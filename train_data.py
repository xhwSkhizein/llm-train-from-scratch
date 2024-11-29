from datasets import load_dataset
from tokenizer import create_tokenizer
import os

def prepare_data(model_type="qwen", tokenizer_path=None):
    """准备训练数据
    
    Args:
        model_type: 使用的tokenizer类型
        tokenizer_path: 自定义tokenizer路径
    """
    # 1. 准备tokenizer
    tokenizer = create_tokenizer(
        model_type=model_type,
        model_path=tokenizer_path
    )

    # 2. 加载数据集
    # dataset = load_dataset("open-phi/textbooks")  # 替换为你的中文数据集
    dataset = load_dataset("mikasenghaas/wikitext-2")  # 替换为你的中文数据集

    # 3. 数据预处理函数
    def tokenize_function(examples):
        # 确保返回的是tensor格式
        return tokenizer(
            examples["content"],  # 使用content字段
            max_length=128,
            padding="max_length",  # 修改padding策略
            truncation=True,
            return_tensors="pt",  # 返回PyTorch tensor
        )

    # 4. 对数据集进行tokenize
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=1000,  # 添加batch_size以提高处理速度
    )

    return tokenized_datasets, tokenizer 
