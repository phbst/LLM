import os
import numpy as np
import torch
import random
from tqdm import tqdm
from tokenizer import Tokenizer

# 配置
DATA_DIR = '/workspace/projects/Bob_llama'
INPUT_FILE = os.path.join(DATA_DIR, 'data/TinyStories_small.txt')
TOKENIZER_MODEL = os.path.join(DATA_DIR, 'tokenizer/tokens_4096.model')
TRAIN_VALID_SPLIT = 0.9  # 90% 训练集, 10% 验证集
MAX_SEQ_LEN = 256  # 或其他你需要的序列长度

def process_data():
    # 创建输出目录
    os.makedirs(os.path.join(DATA_DIR, 'data'), exist_ok=True)
    
    # 加载分词器
    tokenizer = Tokenizer(TOKENIZER_MODEL)
    
    # 读取输入文件
    print(f"从 {INPUT_FILE} 读取数据")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 将内容分割成独立的故事
    stories = content.split('<|endoftext|>')
    stories = [story.strip() for story in stories if story.strip()]
    print(f"找到 {len(stories)} 个故事")
    
    # 随机打乱故事以进行随机分割
    random.seed(42)
    random.shuffle(stories)
    
    # 分割为训练集和验证集
    split_idx = int(len(stories) * TRAIN_VALID_SPLIT)
    train_stories = stories[:split_idx]
    valid_stories = stories[split_idx:]
    
    print(f"训练集: {len(train_stories)} 个故事")
    print(f"验证集: {len(valid_stories)} 个故事")
    
    # 处理并保存训练集
    process_and_save_split(train_stories, tokenizer, "train")
    
    # 处理并保存验证集
    process_and_save_split(valid_stories, tokenizer, "valid")
    
    print("数据处理完成!")

def process_and_save_split(stories, tokenizer, split_name):
    print(f"处理 {split_name} 分割...")
    
    # 创建目录
    split_dir = os.path.join(DATA_DIR, 'data', split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # 对所有故事进行分词
    all_tokens = []
    for story in tqdm(stories):
        # 使用BOS标记进行分词，但不使用EOS标记
        tokens = tokenizer.encode(story, bos=True, eos=False)
        all_tokens.extend(tokens)
    
    # 将tokens转换为numpy数组
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # 将分词数据保存为二进制文件
    bin_file = os.path.join(split_dir, f"{split_name}.bin")
    with open(bin_file, 'wb') as f:
        f.write(all_tokens.tobytes())
    
    # 计算平均序列长度
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum()) if 1 in all_tokens else all_tokens.size
    print(f"保存 {bin_file}, 平均序列长度: {avg_seq_len:.2f}")

class SimpleDataset(torch.utils.data.IterableDataset):
    """用于加载预处理的分词数据的简单数据集"""
    
    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.bin_file = os.path.join('/workspace/projects/Bob_llama/data', split, f"{split}.bin")
        assert os.path.exists(self.bin_file), f"二进制文件 {self.bin_file} 不存在"
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = 42 + worker_id
        rng = random.Random(seed)
        
        # 使用memmap加载所有tokens
        m = np.memmap(self.bin_file, dtype=np.uint16, mode="r")
        
        # 计算完整序列的数量
        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # 移除最后一个可能不完整的批次
        
        assert num_batches > 0, f"文件 {self.bin_file} 对于序列长度 {self.max_seq_len} 来说太小了"
        
        # 生成批次的随机索引
        ixs = list(range(num_batches))
        rng.shuffle(ixs)
        
        # 产生数据批次
        for ix in ixs:
            start = ix * self.max_seq_len
            end = start + self.max_seq_len + 1
            chunk = torch.from_numpy((m[start:end]).astype(np.int64))
            x = chunk[:-1]  # 输入
            y = chunk[1:]   # 目标
            yield x, y

class Task:
    @staticmethod
    def iter_batches(batch_size, split, max_seq_len, device, num_workers=0):
        ds = SimpleDataset(split=split, max_seq_len=max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

if __name__ == "__main__":
    process_data()