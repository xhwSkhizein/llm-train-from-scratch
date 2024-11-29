import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算RoPE位置编码的频率，返回实部和虚部"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]

    # 直接计算 cos 和 sin，而不是使用复数
    freqs_cos = torch.cos(freqs)  # [seq_len, dim/2]
    freqs_sin = torch.sin(freqs)  # [seq_len, dim/2]

    return torch.complex(freqs_cos, freqs_sin)  # 仍然返回复数，但避免中间计算

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """应用RoPE位置编码，不使用复数运算"""
    # 打印输入维度，帮助调试
    # print("xq shape:", xq.shape)
    # print("freqs_cis shape:", freqs_cis.shape)

    # 将输入分割为实部和虚部
    xq_r, xq_i = xq.chunk(2, dim=-1)
    xk_r, xk_i = xk.chunk(2, dim=-1)

    # 获取 freqs_cis 的实部和虚部
    freqs_cos = freqs_cis.real[: xq.size(2)]  # [seq_len, dim/2]
    freqs_sin = freqs_cis.imag[: xq.size(2)]  # [seq_len, dim/2]

    # 调整维度
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim/2]
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim/2]

    # 扩展维度以匹配输入
    freqs_cos = freqs_cos.expand(
        xq.size(0), xq.size(1), -1, -1
    )  # [batch, heads, seq_len, dim/2]
    freqs_sin = freqs_sin.expand(
        xq.size(0), xq.size(1), -1, -1
    )  # [batch, heads, seq_len, dim/2]

    # 应用旋转
    xq_out = torch.cat(
        [xq_r * freqs_cos - xq_i * freqs_sin, xq_i * freqs_cos + xq_r * freqs_sin],
        dim=-1,
    )

    xk_out = torch.cat(
        [xk_r * freqs_cos - xk_i * freqs_sin, xk_i * freqs_cos + xk_r * freqs_sin],
        dim=-1,
    )

    return xq_out, xk_out

class ExpertFFN(nn.Module):
    """单个专家的前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))

class MoEFFN(nn.Module):
    """混合专家FFN层"""
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # 专家们
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # 路由网络
        self.router = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # 计算路由分数
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # 选择top-k个专家
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 准备输出
        output = torch.zeros_like(x_flat)
        
        # 对每个位置应用选中的专家
        for i in range(self.top_k):
            expert_index = top_k_indices[:, i]
            prob = top_k_probs[:, i].unsqueeze(-1)
            
            # 为每个专家收集需要处理的样本
            for j in range(self.num_experts):
                mask = (expert_index == j)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input)
                    output[mask] += prob[mask] * expert_output
        
        return output.view(batch_size, seq_len, d_model)

class FlashSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 为RoPE预计算频率
        self.max_seq_len = 2048  # 可以根据需要调整
        self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len)

    def forward(self, x, mask=None):
        # print(x.shape)  # should be [batch_size, seq_len, d_model]

        # 生成Q,K,V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重排为多头形式，改为 [batch_size, n_heads, seq_len, head_dim]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # 应用RoPE位置编码
        q, k = apply_rotary_emb(q, k, self.freqs_cis.to(q.device))

        # 转换 mask 的类型
        if mask is not None:
            # 将 mask 转换为 float 类型，并设置为 -inf
            mask = mask.to(dtype=q.dtype)  # 确保类型匹配
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(
                -1, self.n_heads, mask.size(-2), mask.size(-1)
            )  # [batch_size, n_heads, seq_len, seq_len]

        # 使用Flash Attention
        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # 回退到普通attention实现
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)

        # 改变输出维度的重排方式
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_moe=False, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.attention = FlashSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 根据配置选择使用MoE还是普通FFN
        if use_moe:
            self.feed_forward = MoEFFN(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力层
        attended = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # 前馈网络(MoE或普通FFN)
        feedforward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feedforward))
        
        return x

class CustomLLM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=512, 
                 n_heads=8, 
                 n_layers=6, 
                 d_ff=2048,
                 use_moe=False,      # 新增MoE开关
                 num_experts=4,
                 top_k=2,
                 max_seq_len=128,
                 dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, 
                n_heads, 
                d_ff,
                use_moe=use_moe,     # 传递MoE配置
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # 输入编码
        # print(input_ids)
        x = self.embedding(input_ids)
        # 应该是 [batch_size, seq_len, d_model]
        # print("Embedding output shape:", x.shape)  # 添加shape检查

        # Transformer层
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits 
