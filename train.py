"""
DeepSeek-V3 Mini Training Script
支持:
- 单卡训练
- 梯度累积
- MTP (Multi-Token Prediction)
- 混合精度训练 (FP16/BF16)
- 梯度裁剪
- 学习率调度
- 真实数据集与分词器支持
"""
import os
import math
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# 导入transformers相关库用于分词器和数据集
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    from datasets import load_dataset, Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/datasets not installed. Install with: pip install transformers datasets")

from model import DeepSeekV3ForCausalLM
from config import DeepSeekV3Config, get_mini_config


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据配置
    train_data_path: Optional[str] = None  # 支持 HuggingFace dataset 名称或本地路径
    eval_data_path: Optional[str] = None
    dataset_name: Optional[str] = None     # HuggingFace dataset 名称，如 "openwebtext"
    dataset_config: Optional[str] = None   # dataset配置
    text_column: str = "text"              # 文本列名
    tokenizer_name_or_path: str = "gpt2"   # 分词器名称或路径
    
    # 训练超参数
    num_epochs: int = 3
    batch_size: int = 1  # 每个设备的batch size
    gradient_accumulation_steps: int = 32  # 梯度累积步数
    max_seq_length: int = 2048
    
    # 优化器参数
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    
    # 混合精度
    mixed_precision: str = "bf16"  # "fp16", "bf16", "no"
    
    # 梯度检查点
    use_gradient_checkpointing: bool = True
    
    # 日志和保存
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # 其他
    seed: int = 42
    num_workers: int = 4
    
    # 有效batch size = batch_size * gradient_accumulation_steps
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class TextDataset(Dataset):
    """
    真实文本数据集，支持本地文件或 HuggingFace datasets
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        text_column: str = "text",
        max_length: int = 2048,
        num_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # 加载数据
        if dataset_name is not None:
            # 从 HuggingFace 加载
            print(f"Loading dataset from HuggingFace: {dataset_name}")
            self.dataset = load_dataset(dataset_name, dataset_config, split="train")
        elif data_path is not None:
            # 从本地加载
            if os.path.isfile(data_path):
                # 单个文件
                print(f"Loading data from file: {data_path}")
                self.dataset = self._load_from_file(data_path)
            elif os.path.isdir(data_path):
                # 目录
                print(f"Loading data from directory: {data_path}")
                self.dataset = self._load_from_directory(data_path)
            else:
                raise ValueError(f"Data path does not exist: {data_path}")
        else:
            raise ValueError("Either dataset_name or data_path must be provided")
            
        # 限制样本数（用于调试）
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            
        print(f"Loaded {len(self.dataset)} samples")
    
    def _load_from_file(self, file_path: str) -> HFDataset:
        """从单个文件加载，支持 .txt, .json, .jsonl"""
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            return load_dataset('json', data_files=file_path, split='train')
        else:
            # 文本文件，每行一个样本
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            return HFDataset.from_dict({self.text_column: texts})
    
    def _load_from_directory(self, dir_path: str) -> HFDataset:
        """从目录加载所有文本文件"""
        all_texts = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(('.txt', '.json', '.jsonl')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.json'):
                                import json
                                data = json.load(f)
                                if isinstance(data, list):
                                    all_texts.extend([str(item) for item in data])
                                else:
                                    all_texts.append(str(data))
                            else:
                                all_texts.extend([line.strip() for line in f if line.strip()])
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return HFDataset.from_dict({self.text_column: all_texts})
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column] if isinstance(item, dict) else item
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",  # 先padding到max_length，DataCollator会处理dynamic padding
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class DataCollatorForCausalLM:
    """
    数据整理器：处理动态padding，并将labels设为input_ids的shift版本
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = False,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm  # 对于 causal LM，mlm=False
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 分离 input_ids 和 attention_mask
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        
        # 找到 batch 中的最大长度（不是整个模型的max_length）
        if self.padding == "max_length":
            max_len = self.max_length
        else:
            max_len = max(len(ids) for ids in input_ids)
            if self.pad_to_multiple_of is not None:
                max_len = ((max_len - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        
        # Padding
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for ids, mask in zip(input_ids, attention_masks):
            length = len(ids)
            # 如果已经在预处理时padding过了，需要截断到实际长度再重新padding
            effective_length = (mask == 1).sum().item() if mask is not None else length
            
            # 获取实际内容（去掉padding）
            actual_ids = ids[:effective_length]
            
            # 对于 Causal LM，labels 就是 input_ids 的 shift
            # 这里先原样复制，loss计算时会 shift
            label_ids = actual_ids.clone()
            
            # Padding 到 batch max length
            padding_length = max_len - effective_length
            
            if padding_length > 0:
                # padding token 在 labels 中设为 -100 (CrossEntropyLoss 的 ignore_index)
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_ids = torch.cat([actual_ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
                padded_mask = torch.cat([torch.ones(effective_length, dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)])
                padded_labels = torch.cat([label_ids, torch.full((padding_length,), -100, dtype=torch.long)])
            else:
                padded_ids = actual_ids[:max_len]
                padded_mask = torch.ones(max_len, dtype=torch.long)
                padded_labels = label_ids[:max_len]
            
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


class DummyDataset(Dataset):
    """虚拟数据集用于测试"""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机序列
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # labels与input_ids相同 (因果语言建模)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn=None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    scheduler_type: str = "cosine",
):
    """获取学习率调度器"""
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, warmup_steps))
        
        if scheduler_type == "constant":
            return 1.0
        
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        elif scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: DeepSeekV3ForCausalLM,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    training_config: TrainingConfig,
    device: torch.device,
    is_accumulation_step: bool,
) -> Dict[str, float]:
    """
    单步训练
    
    Args:
        model: 模型
        batch: 数据批次
        optimizer: 优化器
        scaler: 梯度缩放器 (用于FP16)
        training_config: 训练配置
        device: 设备
        is_accumulation_step: 是否为累积步骤(不是最后一步)
    
    Returns:
        损失字典
    """
    # 将数据移至设备
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # 混合精度前向传播
    device_type = 'cuda' if device.type == 'cuda' else 'cpu' 
    dtype = torch.bfloat16 if training_config.mixed_precision == "bf16" else torch.float16
    
    with autocast(device_type=device_type, enabled=scaler is not None, dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 关键修改：传入 attention_mask
            labels=labels,
            compute_mtp_loss=True,
        )
        loss = outputs["loss"]
        
        # 缩放损失以适应梯度累积
        if training_config.gradient_accumulation_steps > 1:
            loss = loss / training_config.gradient_accumulation_steps
    
    # 反向传播
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # 如果不是累积步骤，则更新权重
    if not is_accumulation_step:
        # 梯度裁剪
        if training_config.max_grad_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config.max_grad_norm
            )
        
        # 优化器步骤
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
    
    # 返回详细损失信息
    result = {
        "loss": loss.item() * training_config.gradient_accumulation_steps,
    }
    
    # 如果有MTP损失，也返回
    if "mtp_losses" in outputs and outputs["mtp_losses"]:
        result["mtp_loss"] = sum(outputs["mtp_losses"]).item() / len(outputs["mtp_losses"])
    
    return result


def evaluate(
    model: DeepSeekV3ForCausalLM,
    eval_dataloader: DataLoader,
    device: torch.device,
    max_eval_steps: Optional[int] = None,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 关键修改：传入 attention_mask
                labels=labels,
                compute_mtp_loss=False,  # 评估时不计算MTP损失
            )
            
            total_loss += outputs["loss"].item()
            total_steps += 1
            
            if max_eval_steps is not None and total_steps >= max_eval_steps:
                break
    
    model.train()
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": math.exp(avg_loss) if avg_loss < 10 else float('inf'),  # 防止溢出
    }


def train(
    model: DeepSeekV3ForCausalLM,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    training_config: TrainingConfig,
    device: torch.device,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """
    训练循环
    
    Args:
        model: 模型
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        training_config: 训练配置
        device: 设备
        tokenizer: 分词器（用于保存）
    """
    # 创建输出目录
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # 保存分词器
    if tokenizer is not None:
        tokenizer.save_pretrained(training_config.output_dir)
        print(f"Tokenizer saved to {training_config.output_dir}")
    
    # 设置随机种子
    torch.manual_seed(training_config.seed)
    
    # 创建数据整理器
    if tokenizer is not None:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            max_length=training_config.max_seq_length,
            padding="longest",  # 动态padding到batch内最长
        )
    else:
        data_collator = None
    
    # 创建数据加载器
    train_dataloader = get_dataloader(
        train_dataset,
        training_config.batch_size,
        collate_fn=data_collator,
        num_workers=training_config.num_workers,
        shuffle=True,
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = get_dataloader(
            eval_dataset,
            training_config.batch_size,
            collate_fn=data_collator,
            num_workers=training_config.num_workers,
            shuffle=False,
        )
    
    # 计算总训练步数
    num_update_steps_per_epoch = len(train_dataloader) // training_config.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * training_config.num_epochs
    
    print(f"训练配置:")
    print(f"  总训练步数: {num_training_steps}")
    print(f"  每epoch更新步数: {num_update_steps_per_epoch}")
    print(f"  有效batch size: {training_config.effective_batch_size}")
    print(f"  设备: {device}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )
    
    # 学习率调度器
    lr_scheduler = get_lr_scheduler(
        optimizer,
        num_training_steps,
        training_config.warmup_steps,
        training_config.lr_scheduler_type,
    )
    
    # 混合精度训练
    scaler = None
    if training_config.mixed_precision == "fp16":
        scaler = GradScaler()
    
    # 训练状态
    global_step = 0
    epoch = 0
    tr_loss = 0.0
    logging_loss = 0.0
    
    # 开始训练
    model.train()
    start_time = time.time()
    
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # 判断是否为累积步骤的最后一步
            is_accumulation_step = (step + 1) % training_config.gradient_accumulation_steps != 0
            
            # 训练步骤
            loss_dict = train_step(
                model,
                batch,
                optimizer,
                scaler,
                training_config,
                device,
                is_accumulation_step,
            )
            
            tr_loss += loss_dict["loss"]
            
            # 如果不是累积步骤，则更新学习率
            if not is_accumulation_step:
                lr_scheduler.step()
                global_step += 1
                
                # 日志
                if global_step % training_config.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / training_config.logging_steps
                    lr = lr_scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    
                    log_msg = (
                        f"Step {global_step}/{num_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.2f}s"
                    )
                    
                    if "mtp_loss" in loss_dict:
                        log_msg += f" | MTP: {loss_dict['mtp_loss']:.4f}"
                    
                    print(log_msg)
                    
                    logging_loss = tr_loss
                
                # 评估
                if eval_dataloader is not None and global_step % training_config.eval_steps == 0:
                    eval_results = evaluate(model, eval_dataloader, device, max_eval_steps=100)
                    print(f"Eval - Loss: {eval_results['eval_loss']:.4f}, PPL: {eval_results['eval_perplexity']:.2f}")
                    model.train()
                
                # 保存检查点
                if global_step % training_config.save_steps == 0:
                    checkpoint_dir = os.path.join(
                        training_config.output_dir,
                        f"checkpoint-{global_step}"
                    )
                    save_checkpoint(model, optimizer, lr_scheduler, global_step, checkpoint_dir, tokenizer)
                    print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Epoch结束
        epoch_loss = tr_loss / (step + 1)
        print(f"Epoch {epoch + 1} finished. Average loss: {epoch_loss:.4f}")
    
    # 保存最终模型
    final_dir = os.path.join(training_config.output_dir, "final")
    save_checkpoint(model, optimizer, lr_scheduler, global_step, final_dir, tokenizer)
    print(f"\nTraining completed! Final model saved to {final_dir}")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 3600:.2f} hours")


def save_checkpoint(
    model: DeepSeekV3ForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    global_step: int,
    output_dir: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型状态
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "pytorch_model.bin")
    )
    
    # 保存优化器和学习率调度器
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": global_step,
        },
        os.path.join(output_dir, "training_state.bin")
    )
    
    # 保存分词器
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    # 保存配置
    # model.config 可以序列化保存
    import json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model.config.__dict__, f, indent=2, default=str)


def load_checkpoint(
    model: DeepSeekV3ForCausalLM,
    checkpoint_dir: str,
    device: torch.device,
) -> DeepSeekV3ForCausalLM:
    """加载检查点"""
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def main():
    """主函数"""
    # 检查依赖
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Please install transformers and datasets: pip install transformers datasets"
        )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 训练配置
    training_config = TrainingConfig(
        # 数据配置示例 - 使用HuggingFace数据集
        # dataset_name="wikitext",
        # dataset_config="wikitext-2-raw-v1",
        # text_column="text",
        
        # 或者使用本地文件/目录
        # train_data_path="./data/train.txt",
        # eval_data_path="./data/val.txt",
        
        tokenizer_name_or_path="gpt2",  # 或其他支持的分词器如 "meta-llama/Llama-2-7b-hf"
        
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=512,
        learning_rate=2e-4,
        warmup_steps=100,
        mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
        use_gradient_checkpointing=True,
        output_dir="./output/deepseek_v3_mini",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
    )
    
    # 加载分词器
    print(f"Loading tokenizer from {training_config.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(training_config.tokenizer_name_or_path)
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 模型配置 - 使用分词器的vocab_size
    model_config = get_mini_config()
    model_config.vocab_size = len(tokenizer)  # 关键：使用分词器的词汇表大小
    model_config.use_gradient_checkpointing = training_config.use_gradient_checkpointing
    
    print(f"模型配置:")
    print(f"  词汇表大小: {model_config.vocab_size}")
    print(f"  层数: {model_config.num_layers}")
    print(f"  隐藏维度: {model_config.hidden_size}")
    print(f"  注意力头数: {model_config.num_attention_heads}")
    
    # 创建模型
    print("\n创建模型...")
    model = DeepSeekV3ForCausalLM(model_config)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 创建数据集
    print("\n创建数据集...")
    
    # 如果有配置真实数据路径，使用真实数据，否则使用DummyDataset
    if training_config.train_data_path is not None or training_config.dataset_name is not None:
        print("使用真实数据集")
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            data_path=training_config.train_data_path,
            dataset_name=training_config.dataset_name,
            dataset_config=training_config.dataset_config,
            text_column=training_config.text_column,
            max_length=training_config.max_seq_length,
            num_samples=50000,  # 可设为None使用全部数据
        )
        eval_dataset = TextDataset(
            tokenizer=tokenizer,
            data_path=training_config.eval_data_path,
            dataset_name=training_config.dataset_name,
            dataset_config=training_config.dataset_config,
            text_column=training_config.text_column,
            max_length=training_config.max_seq_length,
            num_samples=1000,
        ) if training_config.eval_data_path is not None else None
    else:
        print("使用虚拟数据集(DummyDataset)")
        train_dataset = DummyDataset(
            vocab_size=model_config.vocab_size,
            seq_length=training_config.max_seq_length,
            num_samples=1000,
        )
        eval_dataset = DummyDataset(
            vocab_size=model_config.vocab_size,
            seq_length=training_config.max_seq_length,
            num_samples=100,
        )
    
    print(f"\n训练配置:")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.effective_batch_size}")
    print(f"  Mixed precision: {training_config.mixed_precision}")
    
    # 开始训练
    print("\n开始训练...")
    train(model, train_dataset, eval_dataset, training_config, device, tokenizer)


if __name__ == "__main__":
    main()