from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

@dataclass
class SFTDataCollator:
    """
    用于SFT的数据整理器
    """
    tokenizer: object
    max_length: int = 512
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 准备输入输出对
        concatenated_examples = {k: [] for k in features[0].keys()}
        for example in features:
            for k, v in example.items():
                concatenated_examples[k].append(v)
                
        # 格式化为模型输入
        batch = {}
        
        # 处理输入
        input_ids = []
        attention_mask = []
        labels = []
        
        for batch_input_ids, batch_labels in zip(
            concatenated_examples["input_ids"], 
            concatenated_examples["labels"]
        ):
            # 确保长度一致
            combined_length = len(batch_input_ids)
            if combined_length > self.max_length:
                batch_input_ids = batch_input_ids[:self.max_length]
                batch_labels = batch_labels[:self.max_length]
            else:
                # 填充
                padding_length = self.max_length - combined_length
                batch_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                batch_labels.extend([-100] * padding_length)  # -100是PyTorch中忽略的label值
                
            input_ids.append(batch_input_ids)
            attention_mask.append([1] * len(batch_input_ids))
            labels.append(batch_labels)
            
        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        batch["labels"] = torch.tensor(labels)
        
        return batch

class SFTDataset(Dataset):
    """
    用于SFT的数据集
    """
    def __init__(self, tokenizer, data_path=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if data_path:
            self.dataset = load_dataset("json", data_files=data_path)["train"]
        else:
            # 使用示例数据集
            self.dataset = load_dataset("json", data_files={
                "train": "path/to/your/sft_data.json"
            })["train"]
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 构建提示模板
        prompt = f"问：{item['instruction']}\n答："
        response = item['response']
        
        # 编码输入
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        # 构建输入和标签
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + response_ids + [self.tokenizer.eos_token_id]
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def prepare_sft_data(tokenizer, data_path=None, max_length=512):
    """准备SFT数据"""
    dataset = SFTDataset(tokenizer, data_path, max_length)
    data_collator = SFTDataCollator(tokenizer, max_length)
    
    return dataset, data_collator 