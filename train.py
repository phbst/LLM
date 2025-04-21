import json
import math
import os
import time
from contextlib import nullcontext
import torch
from model import Bobllm, ModelArgs
from preprocess_data import Task

# 输出和训练配置
out_dir = "model"  # 模型输出保存路径
eval_interval = 200  # 评估间隔步数
log_interval = 1  # 日志记录间隔步数
eval_iters = 100  # 每次评估时迭代的步数
eval_only = False  # 如果为True，脚本在第一次评估后立即退出
always_save_checkpoint = False  # 如果为True，在每次评估后总是保存检查点
init_from = "scratch"  # 可以选择从头开始训练（'scratch'）或从已有的检查点恢复（'resume'）

# 数据配置
batch_size = 8  # 每个微批次的样本数量
max_seq_len = 256  # 最大序列长度
vocab_size = 4096  # 自定义词汇表大小

# 模型配置
dim = 288  # 模型的隐藏层维度
n_layers = 8  # Transformer的层数
n_heads = 8  # 注意力头的数量
n_kv_heads = 4  # 模型分组
hidden_dim=786 
dropout = 0.0  # Dropout概率

# 优化器配置
gradient_accumulation_steps = 1  # 梯度累积步数，用于模拟更大的批次
learning_rate = 5e-4  # 最大学习率
max_iters = 2000  # 总的训练迭代次数
weight_decay = 1e-1  # 权重衰减系数
beta1 = 0.9  # AdamW优化器的β1参数
beta2 = 0.95  # AdamW优化器的β2参数
grad_clip = 1.0  # 梯度裁剪阈值，0表示不裁剪

# 学习率衰减配置
decay_lr = True  # 是否启用学习率衰减
warmup_iters = 1000  # 学习率预热的步数
lr_decay_iters = max_iters  # 学习率衰减步数
min_lr = 0.0  # 最小学习率

# 系统设置
device = "cuda:0"  # 设备选择
dtype = "bfloat16"  # 数据类型

# 保存配置到字典中，便于日志记录
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}

# 词汇表设置
vocab_source = 'custom'  # 词汇表来源

# 设置随机种子，确保可重复性
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设置设备和数据类型
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = torch.float16
ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

# 创建数据迭代器
def iter_batches(split):
    return Task.iter_batches(
        batch_size=batch_size,
        split=split,
        max_seq_len=max_seq_len,
        device=device,
        num_workers=0,
    )
    #batch_size, split, max_seq_len, device, num_workers=0

# 初始化状态跟踪变量
iter_num = 0
best_val_loss = 1e9

train_loss=[]
valid_loss=[]
# 创建模型参数配置
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

# 初始化模型
model_args = ModelArgs(**model_args)
model = Bobllm(model_args)
model.to(device)

print(f"模型初始化success,device{device}")
# 创建优化器和梯度缩放器
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 损失估计函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["valid"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss
            losses[k] = loss
        out[split] = losses.mean()

    model.train()
    return out

# 学习率调度函数
def get_lr(it):
    # 预热阶段，学习率线性增长
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 若迭代步数超过lr_decay_iters，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 创建保存目录
os.makedirs(out_dir, exist_ok=True)

# 初始化训练数据迭代器
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

# 记录开始时间
t0 = time.time()

# 主训练循环
while True:
    # 更新学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    # 定期评估和保存检查点
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        valid_loss.append({'step':iter_num,'valid_loss':round(losses['valid'].item() , 4)})
        print(f"step {iter_num}:  val loss {losses['valid']:.4f}")
        if losses["valid"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["valid"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                print(f"saving checkpoint to {out_dir}")
        if iter_num == 0 and eval_only:
            break
    
    # 梯度累积训练
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()
    
    # 梯度裁剪和优化器步骤
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # 计时和记录
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        train_loss.append({'step':iter_num,'train_loss':round(lossf, 4)})
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")
    
    # 更新迭代计数器并检查是否达到最大迭代次数
    iter_num += 1
    if iter_num > max_iters:
        break




def save_losses_to_json(loss_data, file_path):
    """
    将损失数据保存到JSON文件
    
    参数:
    loss_data: 包含step和loss信息的字典列表
    file_path: 保存JSON文件的路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 写入JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, ensure_ascii=False, indent=4)
    
    print(f"损失数据已保存到: {file_path}")

save_losses_to_json(train_loss, 'logs/train_losses.json')
save_losses_to_json(valid_loss, 'logs/valid_losses.json')
