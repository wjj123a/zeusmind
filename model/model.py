import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class ZeusMindConfig(PretrainedConfig):
    model_type = "zeusmind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = "silu",
            hidden_size: int = 512,
            intermediate_size: int | None = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000,
            inference_rope_scaling: bool = False,
            flash_attention: bool = True,
            ############ MoE ############
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "softmax",
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            attn_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func
        self.attn_dropout = attn_dropout
        self.residual_dropout = residual_dropout
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 768, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_freqs_cis(
        dim: int = 768,
        end: int = 32 * 1024,
        rope_base: float = 1e6,
        rope_scaling: Optional[dict] = None,
):
    freqs = torch.tensor(1.0) / (
            rope_base ** (torch.arange(0, dim, 2)[: dim // 2] / dim)
    )
    attn_factor = 1.0

    if rope_scaling is not None:
        origin_max = int(rope_scaling["original_max_position_embeddings"])
        factor = float(rope_scaling["factor"])
        beta_fast = float(rope_scaling["beta_fast"])
        beta_slow = float(rope_scaling["beta_slow"])
        attn_factor = float(rope_scaling.get("attention_factor", 1.0))

        if end > origin_max:
            inv_dim = lambda b: (
                                        math.log(origin_max / (2 * math.pi * b)) * dim
                                ) / (2 * math.log(rope_base))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            idx = torch.arange(dim // 2, device=freqs.device, dtype=freqs.dtype)
            ramp = torch.clamp((idx - low) / max(high - low, 0.001), 0, 1)

            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(dim=unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(dim=unsqueeze_dim)).to(q.dtype)
    k_embed = (k * cos.unsqueeze(dim=unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(dim=unsqueeze_dim)).to(k.dtype)

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, kv_head, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[..., None, :].expand(bs, slen, kv_head, n_rep, head_dim).reshape(bs, slen, kv_head * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, config: ZeusMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None \
            else config.num_attention_heads
        assert config.num_attention_heads % self.num_key_value_heads == 0
        assert config.hidden_size % config.num_attention_heads == 0
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attention

    def forward(self, x: torch.Tensor, position_embedding: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bs, slen, _ = x.shape
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:slen], sin[:slen])
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2),
                      repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and slen > 1 and past_key_value is None and (
                attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, is_causal=True,
                                                    dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            scores = (xq @ xk.transpose(-1, -2) / math.sqrt(self.head_dim))
            scores[..., -slen:] += torch.triu(torch.full((slen, slen), float('-inf'), device=scores.device),
                                              diagonal=1).unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask[:, None, None, :scores.shape[-1]]) * -1e9
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        output = output.transpose(1, 2).contiguous().view(bs, slen, -1)
        output = self.residual_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: ZeusMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            self.intermediate_size = 64 * ((int(config.hidden_size * 8 / 3) + 64 - 1) // 64)
        else:
            self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ZeusBlock(nn.Module):
    def __init__(self, layer_id: int, config: ZeusMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                position_embedding: Optional[tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False,
                attention_mask=None):
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            x=self.input_layernorm(hidden_states),
            past_key_value=past_key_value,
            position_embedding=position_embedding,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class ZeusModel(nn.Module):
    def __init__(self, config: ZeusMindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [ZeusBlock(layer_id, config) for layer_id in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(config.hidden_size // config.num_attention_heads,
                                                    config.max_position_embeddings,
                                                    config.rope_theta, config.rope_scaling)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_values: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False,
                **kwargs):
        bs, slen = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * self.num_hidden_layers
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values is not None and past_key_values[0] is not None else 0)
        hidden_states = self.dropout(self.embedding(input_ids))
        position_embedding = self.freqs_cos[start_pos:start_pos + slen], self.freqs_sin[start_pos:start_pos + slen]
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(hidden_states=hidden_states, position_embedding=position_embedding,
                                           past_key_value=past_key_value, use_cache=use_cache,
                                           attention_mask=attention_mask)
            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class ZeusForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ZeusMindConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ZeusMindConfig):
        self.config = config
        super().__init__(config)
        self.model = ZeusModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0, **kwargs):
        hidden_states, present_key_values = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                                       past_key_values=past_key_values,
                                                       use_cache=use_cache, **kwargs)
        if isinstance(logits_to_keep, int):
            logits_hidden_states = (
                hidden_states[:, -logits_to_keep:, :]
                if logits_to_keep > 0
                else hidden_states
            )
        else:
            logits_hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(logits_hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )
