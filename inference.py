"""
DeepSeek-V3 Mini Inference Script
支持:
- KV缓存优化
- 多种解码策略 (贪婪、采样、top-p、top-k)
- 批量推理
"""
import os
import time
from typing import Optional, List, Union, Dict
import torch
import torch.nn.functional as F

from model import DeepSeekV3ForCausalLM
from config import DeepSeekV3Config, get_mini_config, get_tiny_config


class DeepSeekV3Inference:
    """DeepSeek-V3推理类"""
    
    def __init__(
        self,
        model: DeepSeekV3ForCausalLM,
        config: DeepSeekV3Config,
        device: Optional[torch.device] = None,
        use_gradient_checkpointing: bool = True,
    ):
        """
        初始化推理类
        
        Args:
            model: 模型
            config: 配置
            device: 设备
            use_gradient_checkpointing: 保留参数以维持接口兼容，推理中不再使用
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置模型
        self.model = model.to(self.device)
        self.model.eval()
        
        # 梯度检查点已在推理中移除以提升效率
        # 保留参数以维持接口兼容性，但不实际启用
        
        # 词汇表
        self.vocab_size = config.vocab_size
        
        # 特殊token (默认值)
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        early_stopping: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
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
            repetition_penalty: 重复惩罚系数
            do_sample: 是否采样，False则使用贪婪解码
            num_return_sequences: 每个输入返回的序列数
            early_stopping: 是否提前停止 (遇到EOS)
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
            use_cache: 是否使用KV缓存
        
        Returns:
            生成的token IDs [batch_size * num_return_sequences, total_seq_len]
        """
        # 设置特殊token
        pad_token_id = pad_token_id or self.pad_token_id
        eos_token_id = eos_token_id or self.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 如果num_return_sequences > 1，复制输入
        if num_return_sequences > 1:
            input_ids = input_ids.unsqueeze(1).expand(-1, num_return_sequences, -1)
            input_ids = input_ids.reshape(batch_size * num_return_sequences, -1)
        
        # 确定最大生成长度
        input_length = input_ids.shape[1]
        if max_new_tokens is not None:
            max_length = input_length + max_new_tokens
        
        # 初始化
        past_key_values = None if use_cache else [None] * self.config.num_layers
        unfinished_sequences = torch.ones(
            batch_size * num_return_sequences,
            dtype=torch.long,
            device=device
        )
        
        # 记录已生成的token (用于重复惩罚)
        generated_tokens = input_ids.clone()
        
        for step in range(max_length - input_length):
            # 准备输入
            if past_key_values is not None and step > 0:
                # 只使用最后一个token
                model_input_ids = input_ids[:, -1:]
            else:
                model_input_ids = input_ids
            
            # 创建注意力掩码
            attention_mask = torch.ones_like(model_input_ids)
            
            # 前向传播
            outputs = self.model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                compute_mtp_loss=False,
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens,
                    repetition_penalty
                )
            
            # 应用温度
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪解码
            if do_sample:
                next_tokens = self._sample_tokens(next_token_logits, top_p, top_k)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 更新未完成序列
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # 添加到序列
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新KV缓存
            if use_cache:
                past_key_values = outputs["past_key_values"]
            
            # 检查是否完成
            if early_stopping:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(eos_token_id).long()
                )
                
                if unfinished_sequences.max() == 0:
                    break
        
        return input_ids
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚"""
        # 获取已生成token的分数
        score = torch.gather(logits, 1, generated_tokens)
        
        # 如果分数为正，除以惩罚系数；如果为负，乘以惩罚系数
        score = torch.where(score > 0, score / penalty, score * penalty)
        
        # 将修改后的分数放回logits
        logits.scatter_(1, generated_tokens, score)
        
        return logits
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """采样token"""
        batch_size = logits.shape[0]
        
        # Top-k过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for batch_idx in range(batch_size):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                logits[batch_idx, indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        
        # 处理数值不稳定
        probs = torch.nan_to_num(probs, nan=1e-10, posinf=1.0, neginf=0.0)
        probs = probs.clamp(min=1e-10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    @torch.no_grad()
    def batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        **generate_kwargs
    ) -> List[torch.Tensor]:
        """批量生成 (处理不同长度的输入)"""
        results = []
        for input_ids in input_ids_list:
            input_ids = input_ids.to(self.device)
            output = self.generate(input_ids.unsqueeze(0), **generate_kwargs)
            results.append(output[0])
        return results
    
    def get_model_memory_usage(self) -> Dict[str, float]:
        """获取模型显存使用情况"""
        if not torch.cuda.is_available():
            return {"message": "CUDA not available"}
        
        # 模型参数显存
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        # 缓存显存
        cache_memory = 0
        if hasattr(self.model, 'past_key_values'):
            for layer_cache in self.model.past_key_values:
                if layer_cache is not None:
                    for tensor in layer_cache:
                        if tensor is not None:
                            cache_memory += tensor.numel() * tensor.element_size()
        
        return {
            "param_memory_mb": param_memory / (1024 ** 2),
            "cache_memory_mb": cache_memory / (1024 ** 2),
            "total_memory_mb": (param_memory + cache_memory) / (1024 ** 2),
            "allocated_memory_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_memory_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        }


def create_inference_engine(
    checkpoint_path: Optional[str] = None,
    config_name: str = "mini",
    device: Optional[torch.device] = None,
    use_gradient_checkpointing: bool = True,
) -> DeepSeekV3Inference:
    """
    创建推理引擎
    
    Args:
        checkpoint_path: 检查点路径，None则使用随机初始化的模型
        config_name: 配置名称 ("mini"或"tiny")
        device: 设备
        use_gradient_checkpointing: 保留参数以维持接口兼容，推理中不再使用
    
    Returns:
        推理引擎
    """
    # 获取配置
    if config_name == "tiny":
        config = get_tiny_config()
    else:
        config = get_mini_config()
    
    # 设备
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"创建模型 (配置: {config_name})...")
    print(f"  总参数量: {config.total_params / 1e6:.2f}M")
    print(f"  激活参数量: {config.activated_params / 1e6:.2f}M")
    
    # 创建模型
    model = DeepSeekV3ForCausalLM(config)
    
    # 加载检查点
    if checkpoint_path is not None:
        print(f"加载检查点: {checkpoint_path}")
        state_dict = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            map_location=device
        )
        model.load_state_dict(state_dict)
    
    # 创建推理引擎
    engine = DeepSeekV3Inference(
        model=model,
        config=config,
        device=device,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    
    return engine


def demo_generation():
    """生成演示"""
    print("=" * 60)
    print("DeepSeek-V3 Mini 推理演示")
    print("=" * 60)
    
    # 创建推理引擎
    engine = create_inference_engine(
        checkpoint_path=None,  # 使用随机初始化
        config_name="tiny",  # 使用tiny配置以快速测试
        use_gradient_checkpointing=True,
    )
    
    print(f"\n设备: {engine.device}")
    print(f"梯度检查点: {'启用' if engine.model.config.use_gradient_checkpointing else '禁用'}")
    
    # 测试输入 (随机token)
    batch_size = 2
    seq_length = 50
    input_ids = torch.randint(0, engine.vocab_size, (batch_size, seq_length))
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"输入token示例: {input_ids[0, :10].tolist()}")
    
    # 显存使用情况
    memory_info = engine.get_model_memory_usage()
    print(f"\n显存使用情况:")
    for key, value in memory_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value}")
    
    # 贪婪解码
    print("\n" + "-" * 40)
    print("贪婪解码测试")
    print("-" * 40)
    
    start_time = time.time()
    output_greedy = engine.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
    )
    elapsed = time.time() - start_time
    
    print(f"生成序列长度: {output_greedy.shape[1]}")
    print(f"生成token数: {output_greedy.shape[1] - seq_length}")
    print(f"生成时间: {elapsed:.3f}s")
    print(f"生成速度: {(output_greedy.shape[1] - seq_length) / elapsed:.2f} tokens/s")
    print(f"输出示例: {output_greedy[0, -10:].tolist()}")
    
    # 采样解码
    print("\n" + "-" * 40)
    print("采样解码测试 (temperature=0.8, top_p=0.9)")
    print("-" * 40)
    
    start_time = time.time()
    output_sample = engine.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        use_cache=True,
    )
    elapsed = time.time() - start_time
    
    print(f"生成序列长度: {output_sample.shape[1]}")
    print(f"生成token数: {output_sample.shape[1] - seq_length}")
    print(f"生成时间: {elapsed:.3f}s")
    print(f"生成速度: {(output_sample.shape[1] - seq_length) / elapsed:.2f} tokens/s")
    print(f"输出示例: {output_sample[0, -10:].tolist()}")
    
    # 多序列生成
    print("\n" + "-" * 40)
    print("多序列生成测试 (num_return_sequences=3)")
    print("-" * 40)
    
    start_time = time.time()
    output_multi = engine.generate(
        input_ids=input_ids[:1],  # 只用一个输入
        max_new_tokens=15,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=3,
        use_cache=True,
    )
    elapsed = time.time() - start_time
    
    print(f"生成序列数: {output_multi.shape[0]}")
    print(f"每序列token数: {output_multi.shape[1] - seq_length}")
    print(f"总生成token数: {(output_multi.shape[1] - seq_length) * output_multi.shape[0]}")
    print(f"生成时间: {elapsed:.3f}s")
    print(f"生成速度: {(output_multi.shape[1] - seq_length) * output_multi.shape[0] / elapsed:.2f} tokens/s")
    
    # 长序列测试
    print("\n" + "-" * 40)
    print("长序列生成测试 (max_new_tokens=100)")
    print("-" * 40)
    
    long_input = torch.randint(0, engine.vocab_size, (1, 100))
    
    start_time = time.time()
    output_long = engine.generate(
        input_ids=long_input,
        max_new_tokens=100,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        use_cache=True,
    )
    elapsed = time.time() - start_time
    
    print(f"输入长度: {long_input.shape[1]}")
    print(f"输出长度: {output_long.shape[1]}")
    print(f"生成token数: {output_long.shape[1] - long_input.shape[1]}")
    print(f"生成时间: {elapsed:.3f}s")
    print(f"生成速度: {(output_long.shape[1] - long_input.shape[1]) / elapsed:.2f} tokens/s")
    
    # 最终显存使用情况
    print("\n" + "=" * 60)
    print("最终显存使用情况:")
    memory_info = engine.get_model_memory_usage()
    for key, value in memory_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def benchmark_inference():
    """推理性能基准测试"""
    print("=" * 60)
    print("DeepSeek-V3 Mini 推理性能基准测试")
    print("=" * 60)
    
    # 创建推理引擎
    engine = create_inference_engine(
        checkpoint_path=None,
        config_name="mini",  # 使用mini配置
        use_gradient_checkpointing=False,  # 基准测试不使用梯度检查点
    )
    
    # 测试配置
    batch_sizes = [1, 2, 4]
    seq_lengths = [128, 256, 512]
    max_new_tokens = 50
    
    print(f"\n测试配置:")
    print(f"  最大新生成token数: {max_new_tokens}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  序列长度: {seq_lengths}")
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\n测试: batch_size={batch_size}, seq_len={seq_len}")
            
            # 创建输入
            input_ids = torch.randint(0, engine.vocab_size, (batch_size, seq_len))
            
            # 预热
            for _ in range(3):
                _ = engine.generate(
                    input_ids=input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True,
                )
            
            # 正式测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            output = engine.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            
            total_tokens = batch_size * max_new_tokens
            tokens_per_second = total_tokens / elapsed
            
            print(f"  生成时间: {elapsed:.3f}s")
            print(f"  生成速度: {tokens_per_second:.2f} tokens/s")
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "time": elapsed,
                "tokens_per_second": tokens_per_second,
            })
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    for result in results:
        print(f"batch_size={result['batch_size']}, seq_len={result['seq_len']}: "
              f"{result['tokens_per_second']:.2f} tokens/s")


def compare_gradient_checkpointing():
    """比较梯度检查点对性能的影响"""
    print("=" * 60)
    print("梯度检查点功能已在推理中移除")
    print("=" * 60)
    print("说明: 梯度检查点在推理阶段不会带来显存收益，")
    print("      且会增加计算开销，因此已在推理中禁用。")
    print("      建议使用 KV-Cache 进行高效推理。")
    print("=" * 60)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "benchmark":
            benchmark_inference()
        elif command == "compare":
            compare_gradient_checkpointing()
        else:
            print(f"未知命令: {command}")
            print("可用命令: demo, benchmark, compare")
    else:
        demo_generation()


if __name__ == "__main__":
    main()