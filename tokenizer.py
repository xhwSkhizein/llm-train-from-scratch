from transformers import AutoTokenizer
from typing import List, Optional, Union
import os

class TokenizerWrapper:
    """Tokenizer包装器，支持多种开源模型的tokenizer"""

    SUPPORTED_MODELS = {
        "qwen": "Qwen/Qwen2.5-7B",
        "baichuan": "baichuan-inc/Baichuan2-13B-Chat",
        "chatglm": "THUDM/chatglm3-6b",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-v0.1",
        # 可以继续添加其他模型
    }

    def __init__(
        self,
        model_type: str = "qwen",
        model_path: Optional[str] = None,
        trust_remote_code: bool = True,
        use_fast: bool = True
    ):
        """
        初始化tokenizer
        
        Args:
            model_type: 模型类型，支持 'qwen', 'baichuan', 'chatglm', 'llama', 'mistral'
            model_path: 自定义模型路径，如果为None则使用默认路径
            trust_remote_code: 是否信任远程代码
            use_fast: 是否使用fast tokenizer
        """
        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODELS and model_path is None:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types are: {list(self.SUPPORTED_MODELS.keys())}")

        # 确定模型路径
        model_name_or_path = model_path or self.SUPPORTED_MODELS.get(self.model_type)

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast
        )

        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """编码文本"""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ):
        """解码token ids"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def __len__(self):
        """返回词表大小"""
        return len(self.tokenizer)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    def save_pretrained(self, save_directory: str):
        """保存tokenizer"""
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_type_or_path: str,
        trust_remote_code: bool = True,
        use_fast: bool = True,
        **kwargs
    ):
        """从预训练模型加载tokenizer"""
        if model_type_or_path in cls.SUPPORTED_MODELS:
            return cls(model_type=model_type_or_path, 
                     trust_remote_code=trust_remote_code,
                     use_fast=use_fast)
        else:
            return cls(model_path=model_type_or_path,
                     trust_remote_code=trust_remote_code,
                     use_fast=use_fast)

    def get_special_tokens(self):
        """获取特殊token信息"""
        return {
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": self.tokenizer.bos_token,
            "unk_token": self.tokenizer.unk_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }

def create_tokenizer(
    model_type: str = "qwen",
    model_path: Optional[str] = None,
    **kwargs
) -> TokenizerWrapper:
    """
    创建tokenizer的工厂函数
    
    Args:
        model_type: 模型类型
        model_path: 自定义模型路径
        **kwargs: 其他参数传递给TokenizerWrapper
        
    Returns:
        TokenizerWrapper实例
    """
    return TokenizerWrapper(model_type=model_type, model_path=model_path, **kwargs) 
