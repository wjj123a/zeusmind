import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PreTrainedDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(str(sample["text"]), add_special_tokens=False, max_length=self.max_length - 2,
                                         truncation=True)["input_ids"]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask
