from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

@dataclass
class DPODataCollator:
    """DPO数据整理器"""
    tokenizer: object
    max_length: int = 512
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 准备prompt、chosen和rejected回答
        prompts = [f["prompt"] for f in features]
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] for f in features]
        
        # 编码prompt
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码chosen回答
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码rejected回答
        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_tokens.input_ids,
            "prompt_attention_mask": prompt_tokens.attention_mask,
            "chosen_input_ids": chosen_tokens.input_ids,
            "chosen_attention_mask": chosen_tokens.attention_mask,
            "rejected_input_ids": rejected_tokens.input_ids,
            "rejected_attention_mask": rejected_tokens.attention_mask,
        }

class DPODataset(Dataset):
    """DPO数据集"""
    def __init__(self, tokenizer, data_path=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if data_path:
            self.dataset = load_dataset("json", data_files=data_path)["train"]
        else:
            # 使用示例数据集
            self.dataset = load_dataset("json", data_files={
                "train": "path/to/your/dpo_data.json"
            })["train"]
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 构建提示模板
        prompt = f"问：{item['instruction']}\n答："
        
        return {
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }

def prepare_dpo_data(tokenizer, data_path=None, max_length=512):
    """准备DPO数据"""
    dataset = DPODataset(tokenizer, data_path, max_length)
    data_collator = DPODataCollator(tokenizer, max_length)
    
    return dataset, data_collator 