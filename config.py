"""
DeepSeek-V3 Mini Configuration
基于DeepSeek-V3技术报告实现的Mini版本 (1B-1.5B参数)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekV3Config:
    """DeepSeek-V3 Mini配置类"""
    
    # 模型架构参数
    vocab_size: int = 32000  # 词表大小
    num_layers: int = 24  # Transformer层数 (原始61层，缩小到24层)
    hidden_size: int = 1024  # 隐藏层维度 (原始7168，缩小到1024)
    
    # MLA (Multi-head Latent Attention) 参数
    num_attention_heads: int = 16  # 注意力头数 (原始128，按比例缩小)
    qk_head_dim: int = 64  # 每头Q/K维度 (原始128，缩小到64)
    v_head_dim: int = 64  # 每头V维度
    qk_nope_head_dim: int = 64  # 无位置编码的Q/K每头维度
    qk_rope_head_dim: int = 32  # 有RoPE的Q/K每头维度 (原始64，缩小到32)
    kv_lora_rank: int = 128  # KV低秩压缩维度 (原始512，缩小到128)
    q_lora_rank: int = 256  # Query低秩压缩维度 (原始1536，缩小到256)
    
    # RoPE参数
    max_position_embeddings: int = 4096  # 最大位置编码
    rope_theta: float = 10000.0  # RoPE基数
    
    # DeepSeekMoE参数
    # 原始: 1共享专家 + 256路由专家，每个专家中间维度2048
    # 缩小比例: 专家数量和维度都按比例缩小
    num_shared_experts: int = 1  # 共享专家数量
    num_routed_experts: int = 32  # 路由专家数量 (原始256，缩小到32)
    num_experts_per_tok: int = 4  # 每个token激活的专家数 (原始8，缩小到4)
    expert_intermediate_size: int = 256  # 每个专家中间维度 (原始2048，缩小到256)
    
    @property
    def moe_intermediate_size(self) -> int:
        """Dense FFN的中间维度 (前3层使用)"""
        # 与原始DeepSeek-V3保持一致，使用较大的中间维度
        return self.hidden_size * 4  # 4x expansion ratio
    
    # 无辅助损失负载均衡参数
    aux_loss_free: bool = True  # 使用无辅助损失负载均衡
    bias_update_speed: float = 0.001  # 偏置更新速度γ
    seq_aux_loss_factor: float = 0.0001  # 序列级辅助损失系数α
    
    # MTP (Multi-Token Prediction)参数
    num_mtp_tokens: int = 1  # 额外预测的token数量 (D=1表示预测下一个token)
    mtp_loss_weight: float = 0.3  # MTP损失权重
    
    # 归一化参数
    rms_norm_eps: float = 1e-6  # RMSNorm epsilon
    
    # 初始化参数
    initializer_range: float = 0.006  # 初始化标准差
    
    # 训练参数
    dropout_rate: float = 0.0  # Dropout率
    attention_dropout: float = 0.0  # 注意力Dropout
    
    # 梯度检查点
    use_gradient_checkpointing: bool = False  # 是否使用梯度检查点
    
    # 计算属性
    @property
    def head_dim(self) -> int:
        """总头维度 = 无RoPE维度 + 有RoPE维度"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    @property
    def num_key_value_heads(self) -> int:
        """GQA中的key/value头数，MLA中为1"""
        return 1
    
    @property
    def total_params(self) -> int:
        """估算总参数量"""
        # Embedding参数
        embedding_params = self.vocab_size * self.hidden_size
        
        # 每层参数
        # MLA参数
        # Query压缩: hidden_size * q_lora_rank
        # Query升维: q_lora_rank * num_heads * head_dim
        # KV压缩: hidden_size * kv_lora_rank
        # Key升维: kv_lora_rank * num_heads * qk_nope_head_dim
        # Value升维: kv_lora_rank * num_heads * v_head_dim
        # RoPE Key: hidden_size * num_heads * qk_rope_head_dim
        # 输出投影: num_heads * v_head_dim * hidden_size
        mla_params = (
            self.hidden_size * self.q_lora_rank +  # W_DQ
            self.q_lora_rank * self.num_attention_heads * self.head_dim +  # W_UQ
            self.hidden_size * self.kv_lora_rank +  # W_DKV
            self.kv_lora_rank * self.num_attention_heads * self.qk_nope_head_dim +  # W_UK
            self.kv_lora_rank * self.num_attention_heads * self.v_head_dim +  # W_UV
            self.hidden_size * self.num_attention_heads * self.qk_rope_head_dim +  # W_KR
            self.num_attention_heads * self.v_head_dim * self.hidden_size  # W_O
        )
        
        # MoE参数 (假设所有层都使用MoE)
        # 共享专家: 2 * hidden_size * expert_intermediate_size
        # 每个路由专家: 2 * hidden_size * expert_intermediate_size
        # 路由门控: hidden_size * num_routed_experts
        moe_params = (
            2 * self.hidden_size * self.expert_intermediate_size +  # 共享专家
            self.num_routed_experts * 2 * self.hidden_size * self.expert_intermediate_size +  # 路由专家
            self.hidden_size * self.num_routed_experts  # 门控
        )
        
        # 每层总参数
        layer_params = mla_params + moe_params + 2 * self.hidden_size  # 两个RMSNorm
        
        # 所有层
        all_layers_params = self.num_layers * layer_params
        
        # 最终输出层
        output_params = self.hidden_size * self.vocab_size
        
        # MTP模块参数 (每个MTP模块类似一层Transformer)
        mtp_params = self.num_mtp_tokens * (mla_params + moe_params + 2 * self.hidden_size + self.hidden_size * 2 * self.hidden_size)  # 投影矩阵
        
        total = embedding_params + all_layers_params + output_params + mtp_params
        return total
    
    @property
    def activated_params(self) -> int:
        """估算每个token激活的参数量"""
        # Embedding
        embedding_params = self.vocab_size * self.hidden_size
        
        # 每层激活参数
        # MLA参数 (全部激活)
        mla_params = (
            self.hidden_size * self.q_lora_rank +
            self.q_lora_rank * self.num_attention_heads * self.head_dim +
            self.hidden_size * self.kv_lora_rank +
            self.kv_lora_rank * self.num_attention_heads * self.qk_nope_head_dim +
            self.kv_lora_rank * self.num_attention_heads * self.v_head_dim +
            self.hidden_size * self.num_attention_heads * self.qk_rope_head_dim +
            self.num_attention_heads * self.v_head_dim * self.hidden_size
        )
        
        # MoE激活参数
        # 共享专家 + 激活的路由专家
        moe_params = (
            2 * self.hidden_size * self.expert_intermediate_size +  # 共享专家
            self.num_experts_per_tok * 2 * self.hidden_size * self.expert_intermediate_size +  # 激活的路由专家
            self.hidden_size * self.num_routed_experts  # 门控(需要计算所有专家的分数)
        )
        
        layer_params = mla_params + moe_params + 2 * self.hidden_size
        all_layers_params = self.num_layers * layer_params
        output_params = self.hidden_size * self.vocab_size
        
        return embedding_params + all_layers_params + output_params


def get_mini_config() -> DeepSeekV3Config:
    """获取Mini配置 (约1.2B总参数)"""
    return DeepSeekV3Config(
        vocab_size=32000,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        qk_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        kv_lora_rank=128,
        q_lora_rank=256,
        max_position_embeddings=4096,
        num_shared_experts=1,
        num_routed_experts=32,
        num_experts_per_tok=4,
        expert_intermediate_size=256,
        num_mtp_tokens=1,
    )


def get_tiny_config() -> DeepSeekV3Config:
    """获取Tiny配置 (约600M参数，用于快速测试)"""
    return DeepSeekV3Config(
        vocab_size=32000,
        num_layers=16,
        hidden_size=768,
        num_attention_heads=12,
        qk_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        kv_lora_rank=96,
        q_lora_rank=192,
        max_position_embeddings=2048,
        num_shared_experts=1,
        num_routed_experts=16,
        num_experts_per_tok=2,
        expert_intermediate_size=192,
        num_mtp_tokens=1,
    )


# 预设配置
CONFIGS = {
    "mini": get_mini_config(),
    "tiny": get_tiny_config(),
}