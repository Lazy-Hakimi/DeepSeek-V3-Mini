"""
DeepSeek-V3 Mini Model Implementation
完整实现包括:
1. Multi-head Latent Attention (MLA)
2. DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
3. Multi-Token Prediction (MTP)
"""
import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import DeepSeekV3Config


class RMSNorm(nn.Module):
    """RMSNorm归一化层"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 预计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.float32
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # 计算频率
        freqs = torch.outer(t, self.inv_freq)
        # 重复频率以匹配维度
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转张量的一半维度"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    # cos, sin: [seq_len, dim] or [bsz, seq_len, dim]
    # q, k: [bsz, seq_len, num_heads, head_dim]
    
    # 确保cos和sin有正确的维度
    if cos.dim() == 2:
        # [seq_len, dim] -> [1, seq_len, 1, dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.dim() == 3:
        # [bsz, seq_len, dim] -> [bsz, seq_len, 1, dim]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    
    # 确保维度匹配 (处理head_dim可能与dim不同的情况)
    head_dim = q.shape[-1]
    cos_dim = cos.shape[-1]
    if head_dim != cos_dim:
        # 如果维度不匹配，只使用需要的部分
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力 (MLA)
    实现低秩KV压缩以减少推理时的KV缓存
    """
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Query压缩和升维
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        # KV压缩和升维
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        
        # 解耦的Key RoPE投影
        self.k_rope_proj = nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        # RMSNorm用于压缩后的潜在向量
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        MLA前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len]
            position_ids: [batch_size, seq_len]
            past_key_value: 缓存的KV (用于推理)
            use_cache: 是否使用缓存
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Query压缩和升维
        # c_t^Q = W^DQ * h_t
        q_compressed = self.q_down_proj(hidden_states)
        q_compressed = self.q_norm(q_compressed)
        
        # 升维得到无RoPE的Query
        q_nope = self.q_up_proj(q_compressed)  # [bsz, q_len, num_heads * head_dim]
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.head_dim)
        
        # 升维得到带RoPE的Query
        q_rope = self.q_rope_proj(q_compressed)
        q_rope = q_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)
        
        # KV压缩
        # c_t^KV = W^DKV * h_t
        kv_compressed = self.kv_down_proj(hidden_states)
        kv_compressed = self.kv_norm(kv_compressed)
        
        # 升维得到无RoPE的Key和Value
        k_nope = self.k_up_proj(kv_compressed)  # [bsz, q_len, num_heads * qk_nope_head_dim]
        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        
        v = self.v_up_proj(kv_compressed)  # [bsz, q_len, num_heads * v_head_dim]
        v = v.view(bsz, q_len, self.num_heads, self.v_head_dim)
        
        # 解耦的Key RoPE
        k_rope = self.k_rope_proj(hidden_states)  # [bsz, q_len, qk_rope_head_dim]
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [bsz, q_len, num_heads, qk_rope_head_dim]
        
        # 应用RoPE到Query和Key的RoPE部分
        cos, sin = self.rotary_emb(hidden_states, q_len)
        if position_ids is not None:
            # position_ids: [bsz, seq_len]
            # 确保position_ids在有效范围内
            max_pos = cos.size(0)
            position_ids = position_ids.clamp(0, max_pos - 1)
            cos = cos[position_ids]  # [bsz, seq_len, dim]
            sin = sin[position_ids]  # [bsz, seq_len, dim]
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        
        # 拼接Query和Key
        q = torch.cat([q_nope[..., :self.qk_nope_head_dim], q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # 处理缓存 (用于推理)
        if past_key_value is not None:
            past_kv, past_k_rope = past_key_value
            # 只缓存压缩后的KV和RoPE Key
            kv_cache = kv_compressed
            k_rope_cache = k_rope[:, :, 0, :]  # [bsz, q_len, qk_rope_head_dim]
            
            # 合并缓存
            kv_cache = torch.cat([past_kv, kv_cache], dim=1)
            k_rope_cache = torch.cat([past_k_rope, k_rope_cache], dim=1)
            
            # 重新计算Key
            k_nope_cache = self.k_up_proj(kv_cache).view(kv_cache.size(0), -1, self.num_heads, self.qk_nope_head_dim)
            k_rope_cache = k_rope_cache.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope_cache, k_rope_cache], dim=-1)
            
            # 重新计算Value
            v_cache = self.v_up_proj(kv_cache).view(kv_cache.size(0), -1, self.num_heads, self.v_head_dim)
            v = v_cache
            
            if use_cache:
                present_key_value = (kv_cache, k_rope_cache[:, :, 0, :])
            else:
                present_key_value = None
        else:
            if use_cache:
                kv_cache = kv_compressed
                k_rope_cache = k_rope[:, :, 0, :]
                present_key_value = (kv_cache, k_rope_cache)
            else:
                present_key_value = None
        
        # 转置以进行注意力计算
        q = q.transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]
        k = k.transpose(1, 2)  # [bsz, num_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [bsz, num_heads, kv_len, v_head_dim]
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # 应用注意力到Value
        attn_output = torch.matmul(attn_weights, v)  # [bsz, num_heads, q_len, v_head_dim]
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class Expert(nn.Module):
    """单个专家网络 (SwiGLU)"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW1) ⊙ (xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
    """
    
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_intermediate_size = config.expert_intermediate_size
        self.aux_loss_free = config.aux_loss_free
        self.bias_update_speed = config.bias_update_speed
        self.seq_aux_loss_factor = config.seq_aux_loss_factor
        self.layer_idx = layer_idx
        
        # 共享专家
        self.shared_experts = nn.ModuleList([
            Expert(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_shared_experts)
        ])
        
        # 路由专家
        self.routed_experts = nn.ModuleList([
            Expert(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])
        
        # 路由门控
        self.gate = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        
        # 无辅助损失负载均衡的偏置项
        if self.aux_loss_free:
            self.register_buffer("expert_biases", torch.zeros(self.num_routed_experts))
        else:
            self.expert_biases = None
        
        # 用于统计专家负载
        self.register_buffer("expert_loads", torch.zeros(self.num_routed_experts))
        self.register_buffer("num_calls", torch.tensor(0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training_step: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        DeepSeekMoE前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training_step: 是否为训练步骤(用于更新偏置)
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: 辅助损失(仅在训练时)
        """
        bsz, seq_len, _ = hidden_states.size()
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.size(0)
        
        # 共享专家输出
        shared_output = sum(expert(hidden_states_flat) for expert in self.shared_experts)
        
        # 计算路由分数
        router_logits = self.gate(hidden_states_flat)  # [num_tokens, num_routed_experts]
        
        # Sigmoid激活
        affinity_scores = torch.sigmoid(router_logits)  # s_i,t
        
        # 应用偏置进行路由决策 (无辅助损失负载均衡)
        if self.aux_loss_free and self.training:
            routing_scores = affinity_scores + self.expert_biases.unsqueeze(0)
        else:
            routing_scores = affinity_scores
        
        # Top-K路由
        topk_scores, topk_indices = torch.topk(routing_scores, self.num_experts_per_tok, dim=-1)
        
        # 归一化门控值 (使用原始affinity scores，不是加了偏置的)
        topk_affinity = torch.gather(affinity_scores, 1, topk_indices)
        gating_values = topk_affinity / (topk_affinity.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 计算专家输出
        routed_output = torch.zeros_like(hidden_states_flat)
        
        for i in range(self.num_experts_per_tok):
            expert_indices = topk_indices[:, i]
            expert_gates = gating_values[:, i:i+1]
            
            for expert_idx in range(self.num_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.routed_experts[expert_idx](expert_input)
                    routed_output[mask] += expert_gates[mask] * expert_output
        
        # 合并输出
        output = shared_output + routed_output
        output = output.view(bsz, seq_len, self.hidden_size)
        
        # 计算辅助损失
        aux_loss = None
        if self.training:
            # 序列级辅助损失 (防止单个序列内极端不平衡)
            if self.seq_aux_loss_factor > 0:
                aux_loss = self._compute_seq_aux_loss(affinity_scores, topk_indices, seq_len)
            
            # 更新偏置 (无辅助损失负载均衡)
            if self.aux_loss_free and training_step:
                self._update_biases(topk_indices, num_tokens)
        
        return output, aux_loss
    
    def _compute_seq_aux_loss(
        self,
        affinity_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """计算序列级辅助损失"""
        num_tokens = affinity_scores.size(0)
        batch_size = num_tokens // seq_len
        
        # 计算每个token的路由指示
        router_prob = affinity_scores / (affinity_scores.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 计算每个专家的负载
        expert_mask = torch.zeros_like(affinity_scores).scatter_(1, topk_indices, 1.0)
        
        # 按序列计算
        aux_loss = 0.0
        for b in range(batch_size):
            start = b * seq_len
            end = (b + 1) * seq_len
            
            seq_router_prob = router_prob[start:end]  # [seq_len, num_experts]
            seq_expert_mask = expert_mask[start:end]  # [seq_len, num_experts]
            
            # 每个专家的平均亲和度
            f_i = seq_expert_mask.float().mean(dim=0)  # [num_experts]
            # 每个专家的路由概率
            P_i = seq_router_prob.mean(dim=0)  # [num_experts]
            
            # 序列级平衡损失
            seq_aux_loss = self.num_routed_experts * (f_i * P_i).sum()
            aux_loss += seq_aux_loss
        
        aux_loss = aux_loss / batch_size * self.seq_aux_loss_factor
        return aux_loss
    
    def _update_biases(self, topk_indices: torch.Tensor, num_tokens: int):
        """更新专家偏置以实现负载均衡"""
        with torch.no_grad():
            # 计算每个专家的实际负载
            expert_counts = torch.bincount(
                topk_indices.flatten(),
                minlength=self.num_routed_experts
            ).float()
            
            # 期望负载 (均匀分布)
            expected_load = num_tokens * self.num_experts_per_tok / self.num_routed_experts
            
            # 更新偏置: 过载的减少偏置，欠载的增加偏置
            for i in range(self.num_routed_experts):
                if expert_counts[i] > expected_load:
                    self.expert_biases[i] -= self.bias_update_speed
                elif expert_counts[i] < expected_load:
                    self.expert_biases[i] += self.bias_update_speed
            
            # 记录负载统计
            self.expert_loads += expert_counts
            self.num_calls += 1


class DeepSeekV3DecoderLayer(nn.Module):
    """DeepSeek-V3解码器层"""
    
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 自注意力
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(config)
        
        # FFN/MoE (前3层使用Dense FFN，后续使用MoE)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_idx < 3:
            # 前3层使用标准FFN (SwiGLU)
            intermediate_size = config.moe_intermediate_size
            self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # gate
            self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # down
            self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # up
            self.use_moe = False
        else:
            # 后续层使用MoE
            self.mlp = DeepSeekMoE(config, layer_idx)
            self.use_moe = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training_step: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        解码器层前向传播
        """
        residual = hidden_states
        
        # 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # FFN/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states, training_step=training_step)
        else:
            # SwiGLU: Swish(xW1) ⊙ (xW3) W2
            hidden_states = self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))
            aux_loss = None
        
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss


class MTPModule(nn.Module):
    """
    Multi-Token Prediction模块
    预测额外的未来token
    """
    
    def __init__(self, config: DeepSeekV3Config, depth: int):
        super().__init__()
        self.config = config
        self.depth = depth
        
        # 投影矩阵 M_k: 将当前深度表示和下一个token的embedding结合
        # 输入: [h^{k-1}_i; Emb(t_{i+k})]
        self.projection = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        #将RMSNorm作为子模块在初始化时创建
        self.prev_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.next_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Transformer块
        self.layer = DeepSeekV3DecoderLayer(config, layer_idx=config.num_layers + depth)
    
    def forward(
        self,
        prev_hidden_states: torch.Tensor,  # [bsz, seq_len, hidden_size]
        next_token_emb: torch.Tensor,      # [bsz, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MTP模块前向传播
        
        Args:
            prev_hidden_states: 前一深度的表示
            next_token_emb: 下一个token的embedding
            attention_mask: 注意力掩码
            position_ids: 位置ID
        
        Returns:
            hidden_states: 当前深度的表示
        """
        # 归一化输入
        # 使用已初始化的归一化层（参数已与模型在同一设备）
        prev_norm = self.prev_norm(prev_hidden_states)
        next_norm = self.next_norm(next_token_emb)
        
        # 拼接并投影
        combined = torch.cat([prev_norm, next_norm], dim=-1)
        hidden_states = self.projection(combined)
        
        # 通过Transformer块
        hidden_states, _, _ = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            training_step=True,
        )
        
        return hidden_states


class DeepSeekV3PreTrainedModel(nn.Module):
    """DeepSeek-V3预训练模型基类"""
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_layers
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DeepSeekV3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 输出头 (与embedding共享权重)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class DeepSeekV3Model(DeepSeekV3PreTrainedModel):
    """DeepSeek-V3基础模型 (用于推理)"""
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        基础模型前向传播
        """
        batch_size, seq_length = input_ids.shape
        
        # 创建位置ID
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].size(1)
                position_ids = torch.arange(
                    past_length, seq_length + past_length,
                    dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
        
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 创建因果注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=input_ids.device
            )
        
        # 扩展注意力掩码为因果掩码
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds.dtype, past_key_values
        )
        
        # 通过解码器层
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # 梯度检查点 (训练时使用)
            if self.training and self.config.use_gradient_checkpointing:
                layer_outputs = checkpoint(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    False,  # training_step
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    training_step=False,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache.append(layer_outputs[1])
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
            }
        return hidden_states
    
    def _prepare_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype,
        past_key_values: Optional[List] = None,
    ) -> torch.Tensor:
        """准备因果注意力掩码"""
        batch_size, seq_length = input_shape
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=attention_mask.device),
            diagonal=1
        )
        
        # 添加batch维度
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # 结合padding掩码
        if attention_mask is not None:
            # 扩展attention_mask
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
            padding_mask = (1.0 - padding_mask.float()) * float('-inf')
            causal_mask = causal_mask + padding_mask
        
        return causal_mask.to(dtype)


class DeepSeekV3ForCausalLM(DeepSeekV3Model):
    """DeepSeek-V3因果语言模型 (支持MTP训练)"""
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__(config)
        
        # MTP模块
        self.mtp_modules = nn.ModuleList([
            MTPModule(config, depth=i+1)
            for i in range(config.num_mtp_tokens)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        compute_mtp_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        因果语言模型前向传播 (支持MTP)
        """
        batch_size, seq_length = input_ids.shape
        
        # 基础模型前向传播
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,  # 需要hidden states用于MTP
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"]
        all_hidden_states = outputs["hidden_states"]
        
        # 主模型预测 (下一个token)
        logits = self.lm_head(hidden_states)
        
        # 计算主损失
        loss = None
        mtp_losses = []
        
        if labels is not None:
            # 主损失 - 使用ignore_index处理padding
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 关键修改：忽略-100的padding
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # MTP损失
            if compute_mtp_loss and self.config.num_mtp_tokens > 0:
                for mtp_idx, mtp_module in enumerate(self.mtp_modules):
                    # 获取目标token的embedding
                    target_positions = mtp_idx + 2  # 预测第mtp_idx+2个token
                    if target_positions <= seq_length:
                        target_token_ids = labels[..., target_positions-1:].contiguous()
                        
                        # 获取当前深度的hidden states
                        if mtp_idx == 0:
                            prev_hidden = hidden_states
                        else:
                            prev_hidden = mtp_hidden
                        
                        # 获取下一个token的embedding
                        # 注意：对于padding tokens (-100)，我们使用0作为embedding索引（会被mask忽略）
                        safe_token_ids = target_token_ids.clone()
                        safe_token_ids[safe_token_ids == -100] = 0  # 临时替换，避免embedding报错
                        next_token_emb = self.embed_tokens(safe_token_ids)
                        
                        # 截断到相同长度
                        min_len = min(prev_hidden.size(1), next_token_emb.size(1))
                        prev_hidden_trunc = prev_hidden[:, :min_len, :]
                        next_token_emb_trunc = next_token_emb[:, :min_len, :]
                        
                        # MTP模块前向传播 - 使用原始模型的attention_mask
                        # 注意：需要调整mask以匹配截断后的长度
                        if attention_mask is not None:
                            mtp_attention_mask = attention_mask[:, :min_len]
                        else:
                            mtp_attention_mask = None
                            
                        mtp_hidden = mtp_module(
                            prev_hidden_states=prev_hidden_trunc,
                            next_token_emb=next_token_emb_trunc,
                            attention_mask=mtp_attention_mask,
                            position_ids=position_ids[:, :min_len] if position_ids is not None else None,
                        )
                        
                        # MTP预测
                        mtp_logits = self.lm_head(mtp_hidden)
                        
                        # MTP损失 - 同样使用ignore_index
                        mtp_shift_logits = mtp_logits[..., :-1, :].contiguous()
                        mtp_shift_labels = target_token_ids[..., 1:].contiguous()
                        
                        if mtp_shift_logits.numel() > 0:
                            mtp_loss = loss_fct(
                                mtp_shift_logits.view(-1, self.config.vocab_size),
                                mtp_shift_labels.view(-1)
                            )
                            mtp_losses.append(mtp_loss)
                
                # 合并MTP损失
                if mtp_losses:
                    avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)
                    loss = loss + self.config.mtp_loss_weight * avg_mtp_loss
        
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "past_key_values": outputs["past_key_values"],
                "hidden_states": all_hidden_states,
                "mtp_losses": mtp_losses,
            }
        return logits
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """为生成准备输入"""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            max_length: 最大序列长度 (包括输入)
            max_new_tokens: 最大新生成token数 (优先级高于max_length)
            temperature: 采样温度
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            do_sample: 是否采样
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
        
        Returns:
            生成的token IDs
        """
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 2
        
        # 如果num_return_sequences > 1，复制输入
        if num_return_sequences > 1:
            input_ids = input_ids.unsqueeze(1).expand(-1, num_return_sequences, -1)
            input_ids = input_ids.reshape(batch_size * num_return_sequences, -1)
            batch_size = batch_size * num_return_sequences
        
        # 确定最大生成长度
        if max_new_tokens is not None:
            max_length = input_length + max_new_tokens
        elif max_length is None:
            max_length = input_length + 100  # 默认生成100个token
        
        # 初始化
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_length - input_length):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past_key_values, None
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs, compute_mtp_loss=False)
            
            # 获取下一个token的logits
            next_token_logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样
            if do_sample:
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 处理数值不稳定
                probs = torch.nan_to_num(probs, nan=1e-10, posinf=1.0, neginf=0.0)
                probs = probs.clamp(min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 更新未完成序列
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # 更新输入
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新past_key_values
            past_key_values = outputs["past_key_values"]
            
            # 检查是否完成
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(eos_token_id).long()
            )
            
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids