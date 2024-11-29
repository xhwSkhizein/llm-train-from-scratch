# LLM Training from scratch on a single RTX 3090

一个用于训练中文大语言模型的demo代码, 支持预训练、SFT(Supervised Fine-tuning)和 DPO(Direct Preference Optimization)训练


## 安装

```bash
# 安装基础依赖
pip install torch transformers datasets accelerate wandb einops
# 克隆仓库
git clone https://github.com/xhwSkhizein/llm-train-from-scratch
cd llm-train-from-scratch
```

## 使用方法

### 1. 数据准备

#### 预训练数据

```python
from train_data import prepare_data
# 准备与训练数据
tokenized_datasets, tokenizer = prepare_data(
    tokenizer_path="path/to/tokenizer" # 可选
)
```

#### SFT 数据

```json
[
  {
    "instruction": "请介绍一下你自己",
    "response": "我是一个AI助手，我可以帮助回答问题、解决问题，和进行友好的对话。"
  }
]
```

#### DPO 数据

```json
[
  {
    "instruction": "请介绍一下你自己",
    "chosen": "我是一个AI助手，我会尊重用户并提供有帮助的回答。",
    "rejected": "我是最强大的AI，我什么都知道。"
  }
]
```

### 2. 训练

#### 预训练

```bash
python train_llm.py
```

#### SFT 训练

```bash
python sft_train.py
```

#### DPO 训练

```bash
python dpo_train.py
```

### 3. 配置

所有训练脚本都支持通过 wandb_config 配置训练参数：

```python
wandb_config = {
    "d_model": 512, # 模型维度
    "n_heads": 8, # 注意力头数
    "n_layers": 6, # 层数
    "d_ff": 2048, # 前馈网络维度
    "batch_size": 32, # 批次大小
    "learning_rate": 5e-4, # 学习率
    "num_epochs": 3, # 训练轮数
    "use_moe": False, # 是否使用MoE
    # ... 其他参数
}
```

## 项目结构

```
chinese-llm-training/
├── models.py # 模型定义
├── tokenizer.py # 分词器实现
├── train_data.py # 预训练数据处理
├── train_llm.py # 预训练脚本
├── sft_data.py # SFT数据处理
├── sft_train.py # SFT训练脚本
├── dpo_data.py # DPO数据处理
├── dpo_train.py # DPO训练脚本
└── README.md
```

## 特性

### 1. 模型架构

- 基于 Transformer 的自定义 LLM 实现
- 支持 RoPE(Rotary Position Embedding)位置编码
- 可选的 MoE(Mixture of Experts)前馈网络
- 支持 Flash Attention 加速
- 支持梯度检查点(Gradient Checkpointing)

### 2. 训练方法

- 预训练(Pre-training)
- 有监督微调(SFT)
- 直接偏好优化(DPO)

### 3. 性能优化

- Flash Attention 加速
- 混合精度训练(FP16)
- 梯度累积
- 梯度检查点
- torch.compile 加速
- 多进程数据加载优化

### 4. 监控与可视化

- wandb 集成
- 训练损失追踪
- GPU 内存监控
- 实时生成示例
- 训练进度记录


## 主要组件

### CustomLLM

- 支持 RoPE 位置编码
- 可选的 MoE 层
- Flash Attention 支持
- 灵活的配置选项


### 训练优化

- 混合精度训练
- 梯度累积
- 梯度检查点
- Flash Attention
- torch.compile 加速

## 性能建议

1. **显存优化**:

   - 使用梯度检查点
   - 启用混合精度训练
   - 适当的批次大小
   - 合理的梯度累积步数

2. **训练速度**:

   - 启用 Flash Attention
   - 使用 torch.compile
   - 优化数据加载
   - 调整 worker 数量

3. **模型选择**:
   - 根据实际需求选择模型大小
   - 考虑是否启用 MoE
   - 权衡性能和效果

## 注意事项

1. 确保 CUDA 版本与 PyTorch 匹配
2. 预训练前检查数据质量
3. SFT 和 DPO 训练需要高质量的数据
4. 根据 GPU 显存调整配置参数
5. 定期保存检查点

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
