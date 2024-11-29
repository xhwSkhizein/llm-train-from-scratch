import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from models import CustomLLM
from dpo_data import prepare_dpo_data
import torch.nn.functional as F

class DPOLoss(torch.nn.Module):
    """DPO损失函数"""
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps):
        """计算DPO损失"""
        # 计算logits差异
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        # 计算损失
        advantages = chosen_rewards - rejected_rewards
        loss = -torch.nn.functional.logsigmoid(advantages / self.beta).mean()
        
        return loss

def train_dpo(
    project_name="chinese-llm-dpo",
    run_name="dpo-model",
    wandb_config=None,
    data_path=None,
    reference_model_path=None
):
    # 1. 初始化wandb
    if wandb_config is None:
        wandb_config = {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "batch_size": 8,  # DPO通常需要更小的batch size
            "learning_rate": 1e-6,  # DPO使用更小的学习率
            "num_epochs": 3,
            "use_moe": False,
            "max_length": 512,
            "beta": 0.1  # DPO温度参数
        }
    
    wandb.init(project=project_name, name=run_name, config=wandb_config)
    
    # 2. 初始化accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=8,  # DPO需要更多的梯度累积
        log_with="wandb"
    )
    
    # 3. 准备数据
    dataset, data_collator = prepare_dpo_data(
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
    # 策略模型（要训练的模型）
    policy_model = CustomLLM(
        vocab_size=len(tokenizer),
        d_model=wandb.config.d_model,
        n_heads=wandb.config.n_heads,
        n_layers=wandb.config.n_layers,
        d_ff=wandb.config.d_ff,
        use_moe=wandb.config.use_moe,
        max_seq_len=wandb.config.max_length
    )
    
    # 参考模型（固定权重）
    reference_model = CustomLLM(
        vocab_size=len(tokenizer),
        d_model=wandb.config.d_model,
        n_heads=wandb.config.n_heads,
        n_layers=wandb.config.n_layers,
        d_ff=wandb.config.d_ff,
        use_moe=wandb.config.use_moe,
        max_seq_len=wandb.config.max_length
    )
    
    # 加载参考模型权重
    if reference_model_path:
        reference_model.load_state_dict(torch.load(reference_model_path))
    reference_model.eval()  # 设置为评估模式
    
    # 5. 初始化DPO损失
    dpo_loss = DPOLoss(beta=wandb.config.beta)
    
    # 6. 优化器设置
    optimizer = AdamW(
        policy_model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=0.01
    )
    
    # 7. 使用torch.compile加速
    if hasattr(torch, 'compile'):
        policy_model = torch.compile(policy_model)
        reference_model = torch.compile(reference_model)
    
    # 8. 准备训练
    policy_model, optimizer, train_dataloader = accelerator.prepare(
        policy_model, optimizer, train_dataloader
    )
    reference_model = accelerator.prepare(reference_model)
    
    # 9. 训练循环
    policy_model.train()
    completed_steps = 0
    
    for epoch in range(wandb.config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy_model):
                # 获取策略模型的logprobs
                policy_chosen_outputs = policy_model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                policy_rejected_outputs = policy_model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )
                
                # 获取参考模型的logprobs
                with torch.no_grad():
                    reference_chosen_outputs = reference_model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"]
                    )
                    reference_rejected_outputs = reference_model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"]
                    )
                
                # 计算损失
                loss = dpo_loss(
                    policy_chosen_outputs,
                    policy_rejected_outputs,
                    reference_chosen_outputs,
                    reference_rejected_outputs
                )
                
                # 反向传播
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(policy_model.parameters(), 1.0)
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
                    
                    if accelerator.is_main_process:
                        policy_model.eval()
                        with torch.no_grad():
                            example_prompt = "问：请介绍一下你自己\n答："
                            input_ids = tokenizer.encode(example_prompt, return_tensors="pt").to(policy_model.device)
                            outputs = policy_model.generate(
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
                        policy_model.train()
        
        # 保存模型
        accelerator.save_state(f"./checkpoints/dpo_epoch_{epoch}")
    
    # 10. 结束训练
    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__":
    wandb_config = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "batch_size": 8,
        "learning_rate": 1e-6,
        "num_epochs": 3,
        "use_moe": False,
        "max_length": 512,
        "beta": 0.1,
        "gradient_accumulation_steps": 8,
        "mixed_precision": "fp16",
    }
    
    train_dpo(
        project_name="chinese-llm-dpo",
        run_name="dpo-model-v1",
        wandb_config=wandb_config,
        data_path="path/to/your/dpo_data.json",
        reference_model_path="path/to/your/reference_model.pth"
    ) 