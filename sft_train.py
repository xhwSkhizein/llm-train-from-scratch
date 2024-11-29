import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from models import CustomLLM
from sft_data import prepare_sft_data
import torch.nn.functional as F

def train_sft(
    project_name="chinese-llm-sft",
    run_name="sft-model",
    wandb_config=None,
    data_path=None
):
    # 1. 初始化wandb
    if wandb_config is None:
        wandb_config = {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "batch_size": 16,  # SFT通常使用较小的batch size
            "learning_rate": 1e-5,  # SFT通常使用较小的学习率
            "num_epochs": 3,
            "use_moe": False,
            "max_length": 512
        }
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config
    )
    
    # 2. 初始化accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=4,  # SFT通常需要更多的梯度累积
        log_with="wandb"
    )
    
    # 3. 准备数据
    dataset, data_collator = prepare_sft_data(
        tokenizer,
        data_path,
        max_length=wandb.config.max_length
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )
    
    # 4. 初始化模型
    model = CustomLLM(
        vocab_size=len(tokenizer),
        d_model=wandb.config.d_model,
        n_heads=wandb.config.n_heads,
        n_layers=wandb.config.n_layers,
        d_ff=wandb.config.d_ff,
        use_moe=wandb.config.use_moe,
        max_seq_len=wandb.config.max_length
    )
    
    # 5. 优化器设置
    optimizer = AdamW(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=0.01  # SFT通常需要更强的正则化
    )
    
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
                
                # 计算损失 (只计算非padding和非prompt部分的loss)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100  # 忽略padding和prompt部分
                )
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪 (SFT通常需要梯度裁剪)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                completed_steps += 1
                
                # 记录指标
                if step % 10 == 0:
                    accelerator.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
                
                # 定期生成示例
                if step % 100 == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                    
                    # 生成一个示例
                    if accelerator.is_main_process:
                        model.eval()
                        with torch.no_grad():
                            example_prompt = "问：请介绍一下你自己\n答："
                            input_ids = tokenizer.encode(example_prompt, return_tensors="pt").to(model.device)
                            outputs = model.generate(
                                input_ids,
                                max_length=100,
                                num_return_sequences=1,
                                temperature=0.7
                            )
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            accelerator.log(
                                {"example_generation": generated_text},
                                step=completed_steps
                            )
                        model.train()
                    
                    # 记录GPU使用情况
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        accelerator.log(
                            {"gpu_memory_MB": gpu_memory},
                            step=completed_steps
                        )
        
        # 每个epoch结束后保存模型
        accelerator.save_state(f"./checkpoints/sft_epoch_{epoch}")
        
        # 记录epoch级别的指标
        accelerator.log(
            {
                "epoch": epoch,
                "epoch_loss": loss.item(),
            },
            step=completed_steps
        )
    
    # 10. 结束训练
    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__":
    wandb_config = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "use_moe": False,
        "max_length": 512,
        "gradient_accumulation_steps": 4,
        "mixed_precision": "fp16",
    }
    
    train_sft(
        project_name="chinese-llm-sft",
        run_name="sft-model-v1",
        wandb_config=wandb_config,
        data_path="path/to/your/sft_data.json"  # 你的SFT数据路径
    ) 