from datasets import load_dataset
from tokenizer import create_tokenizer
import torch
import os
import json


def prepare_data(model_type="qwen", tokenizer_path=None, use_cache=False):
    """准备训练数据

    Args:
        model_type: 使用的tokenizer类型
        tokenizer_path: 自定义tokenizer路径
        use_cache: 是否使用缓存的数据
    """
    # 定义缓存路径
    cache_dir = "./cache_data"
    cache_file = os.path.join(cache_dir, f"{model_type}_tokenized_data.cache")

    # 如果使用缓存且缓存文件存在，直接加载缓存
    if use_cache and os.path.exists(cache_file):
        print(f"从缓存加载数据: {cache_file}")
        tokenized_datasets = load_dataset("json", data_files=cache_file)
        tokenizer = create_tokenizer(model_type=model_type, model_path=tokenizer_path)
        return tokenized_datasets, tokenizer

    # 1. 准备tokenizer
    tokenizer = create_tokenizer(
        model_type=model_type,
        model_path=tokenizer_path
    )

    # 2. 加载数据集
    dataset = load_dataset("mikasenghaas/wikitext-2")
    print(dataset["train"].column_names)

    # 3. 数据预处理函数
    def tokenize_function(examples):
        outputs = tokenizer.encode(
            examples["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # 4. 对数据集进行tokenize
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 如果需要缓存，保存处理后的数据
    if use_cache:
        print(f"保存数据到缓存: {cache_file}")
        os.makedirs(cache_dir, exist_ok=True)
        # 将数据集保存为JSON格式
        tokenized_datasets.save_to_disk(cache_file)

    return tokenized_datasets, tokenizer
