"""
DST ProAct Configuration

Configuration classes for the ProAssist-style DST model.
"""

from enum import Enum
from typing import Optional
from transformers import PretrainedConfig, LlamaConfig


class ExceedContextHandling(Enum):
    """Strategy for handling context that exceeds max_seq_len."""
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"


class DSTProActConfig(PretrainedConfig):
    """
    Configuration for DST-extended ProAct model.
    
    Attributes:
        llm_pretrained: Base LLM model name/path
        vision_hidden_size: Dimension of pre-computed vision embeddings
        max_seq_len: Maximum sequence length for training
        img_token: Token used to mark image positions
        use_speaking_decision_head: Enable speaking decision head
        use_dst_update_head: Enable DST update decision head
        binary_loss_weight: Weight for binary classification losses
    """
    
    model_type = "dst_proact"
    
    def __init__(
        self,
        *,
        llm_pretrained: str = "meta-llama/Llama-3.2-3B-Instruct",
        vision_hidden_size: int = 1152,
        max_seq_len: int = 4096,
        padding_side: str = "left",
        ignore_id: int = -100,
        img_token: str = "<image>",
        img_token_id: Optional[int] = None,
        dst_gen_token: str = "[DST]",
        dst_gen_token_id: Optional[int] = None,
        asst_gen_token: str = "[ASST]",
        asst_gen_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_speaking_decision_head: bool = True,
        use_dst_update_head: bool = True,
        binary_decision_head_type: str = "linear",
        binary_loss_weight: float = 1.0,
        exceed_context_handling: str = "drop_all",
        attn_implementation: Optional[str] = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.llm_pretrained = llm_pretrained
        self.vision_hidden_size = vision_hidden_size
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side
        self.ignore_id = ignore_id
        self.img_token = img_token
        self.img_token_id = img_token_id
        self.dst_gen_token = dst_gen_token
        self.dst_gen_token_id = dst_gen_token_id
        self.asst_gen_token = asst_gen_token
        self.asst_gen_token_id = asst_gen_token_id
        self.eos_token_id = eos_token_id
        self.use_speaking_decision_head = use_speaking_decision_head
        self.use_dst_update_head = use_dst_update_head
        self.binary_decision_head_type = binary_decision_head_type
        self.binary_loss_weight = binary_loss_weight
        self.attn_implementation = attn_implementation
        
        if exceed_context_handling not in ExceedContextHandling._value2member_map_:
            raise ValueError(f"Unsupported exceed_context_handling: {exceed_context_handling}")
        self.exceed_context_handling = exceed_context_handling
    
    @property
    def exceed_context_handling_strategy(self) -> ExceedContextHandling:
        return ExceedContextHandling(self.exceed_context_handling)


class DSTProActLlamaConfig(LlamaConfig, DSTProActConfig):
    """
    Combined Llama + DST ProAct configuration.
    
    Usage:
        config = DSTProActLlamaConfig.from_pretrained_llama(
            "meta-llama/Llama-3.2-3B-Instruct",
            vision_hidden_size=1152,
        )
    """
    
    model_type = "dst_proact_llama"
    
    # DST-specific default values
    _dst_defaults = {
        'llm_pretrained': "meta-llama/Llama-3.2-3B-Instruct",
        'vision_hidden_size': 1152,
        'max_seq_len': 4096,
        'padding_side': "left",
        'ignore_id': -100,
        'img_token': "<image>",
        'img_token_id': None,
        'dst_gen_token': "[DST]",
        'dst_gen_token_id': None,
        'asst_gen_token': "[ASST]",
        'asst_gen_token_id': None,
        'use_speaking_decision_head': True,
        'use_dst_update_head': True,
        'binary_decision_head_type': "linear",
        'binary_loss_weight': 1.0,
        'exceed_context_handling': "drop_all",
        'attn_implementation': "flash_attention_2",
    }
    
    def __init__(self, **kwargs):
        # Set DST defaults
        for key, default in self._dst_defaults.items():
            if key not in kwargs:
                kwargs[key] = default
        
        # Extract DST kwargs
        dst_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self._dst_defaults}
        
        LlamaConfig.__init__(self, **kwargs)
        
        for key, value in dst_kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_pretrained_llama(cls, llm_pretrained: str, **kwargs) -> "DSTProActLlamaConfig":
        """Create config from pretrained Llama model."""
        llama_config = LlamaConfig.from_pretrained(llm_pretrained)
        merged_config = llama_config.to_dict()
        merged_config.update(kwargs)
        merged_config['llm_pretrained'] = llm_pretrained
        return cls(**merged_config)
