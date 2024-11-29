import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from models import CustomLLM
from train_data import prepare_data

def train(
    project_name="chinese-llm",
    run_name="base-model",
    wandb_config=None
):
    # 1. 初始化wandb
    if wandb_config is None:
        wandb_config = {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "num_epochs": 3,
            "use_moe": False
        }
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config
    )
    
    # 2. 初始化accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',    # 启用混合精度训练
        gradient_accumulation_steps=2,  # 梯度累积
        log_with="wandb"  # 设置使用wandb记录日志
    )
    
    # 3. 准备数据
    tokenized_datasets, tokenizer = prepare_data()
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=wandb.config.batch_size,
        shuffle=True,
        pin_memory=True,  # 数据加载优化
        num_workers=4     # 多进程数据加载
    )
    
    # 4. 初始化模型
    model = CustomLLM(
        vocab_size=len(tokenizer),
        d_model=wandb.config.d_model,
        n_heads=wandb.config.n_heads,
        n_layers=wandb.config.n_layers,
        d_ff=wandb.config.d_ff,
        use_moe=wandb.config.use_moe,
        max_seq_len=128
    )
    
    # 5. 优化器设置
    optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
    
    # 6. 使用torch.compile加速
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # 7. 准备训练
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # 8. 启用梯度检查点
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # 9. 训练循环
    model.train()
    total_steps = len(train_dataloader) * wandb.config.num_epochs
    progress_bar = accelerator.init_trackers(
        project_name=project_name,
        config=wandb.config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    
    completed_steps = 0
    
    for epoch in range(wandb.config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # 计算损失
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    batch["input_ids"].view(-1)
                )
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                completed_steps += 1
                
                # 记录指标
                if step % 10 == 0:  # 每10步记录一次
                    accelerator.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
                
                if step % 100 == 0:  # 每100步打印一次
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                    
                    # 记录GPU使用情况
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        accelerator.log(
                            {"gpu_memory_MB": gpu_memory},
                            step=completed_steps
                        )
        
        # 每个epoch结束后保存模型和记录验证指标
        accelerator.save_state(f"./checkpoints/epoch_{epoch}")
        
        # 记录epoch级别的指标
        accelerator.log(
            {
                "epoch": epoch,
                "epoch_loss": loss.item(),  # 使用最后一个batch的loss作为epoch loss
            },
            step=completed_steps
        )
    
    # 10. 结束训练，关闭wandb
    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__":
    # 可以通过命令行参数或配置文件传入这些参数
    wandb_config = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "num_epochs": 3,
        "use_moe": False,
        "gradient_accumulation_steps": 2,
        "mixed_precision": "fp16",
    }
    
    train(
        project_name="chinese-llm",
        run_name="base-model-v1",
        wandb_config=wandb_config
    ) 