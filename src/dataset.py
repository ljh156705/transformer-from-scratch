import torch
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import io
import os
from collections import Counter

class Vocabulary:
    """词汇表类，用于管理词到索引的映射"""
    def __init__(self, counter, specials):
        self.specials = specials
        self.itos = list(specials)
        # 按频率降序排序
        for word, _ in counter.most_common():
            if word not in self.itos:
                self.itos.append(word)
        
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.unk_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def get_index(self, word):
        return self.stoi.get(word, self.unk_index)

    def get_word(self, index):
        return self.itos[index] if 0 <= index < len(self.itos) else "<unk>"

def download_and_extract_wikitext2(data_path="."):
    """
    下载并解压 WikiText-2 数据集。
    """
    # 更新为新的下载链接
    url = "https://github.com/LogSSim/deeplearning_d2l_classes/raw/main/class14_BERT/wikitext-2-v1.zip"
    zip_path = os.path.join(data_path, "wikitext-2-v1.zip")
    extracted_path = os.path.join(data_path, "wikitext-2")

    if os.path.exists(extracted_path):
        print("数据集已存在，跳过下载和解压。")
        return extracted_path

    print("正在下载 WikiText-2 数据集...")
    try:
        r = requests.get(url)
        r.raise_for_status() # 如果下载失败则抛出异常
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print("正在解压数据集...")
        z.extractall(data_path)
        print(f"数据集已成功解压到: {extracted_path}")
        return extracted_path
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"解压失败: {e}. 下载的文件可能不是一个有效的zip文件。")
        return None

def build_vocab_from_file(filepath):
    """
    从文件构建词汇表。
    """
    counter = Counter()
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            # 过滤掉空行和标题行
            if line.strip() and not line.strip().startswith("="):
                counter.update(line.strip().split())
    
    vocab = Vocabulary(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    return vocab

class LanguageModelDataset(Dataset):
    """
    自定义语言模型数据集。
    """
    def __init__(self, data_tensor, bptt):
        self.data = data_tensor
        self.bptt = bptt

    def __len__(self):
        # 确保我们不会因为目标序列而出界
        return (len(self.data) - 1) // self.bptt

    def __getitem__(self, idx):
        start_idx = idx * self.bptt
        end_idx = start_idx + self.bptt
        src = self.data[start_idx:end_idx]
        # 目标是源序列向右移动一位
        tgt = self.data[start_idx + 1 : end_idx + 1]
        return src, tgt

def tokenize_file(filepath, vocab):
    """
    将文件中的文本转换为 token ID 的长序列。
    """
    token_ids = []
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("="):
                tokens = ["<bos>"] + line.strip().split() + ["<eos>"]
                token_ids.extend([vocab.get_index(token) for token in tokens])
    return torch.tensor(token_ids, dtype=torch.long)


def get_data_loaders(batch_size, bptt=35):
    """
    主函数，用于获取数据加载器。
    """
    print("--- 开始数据处理 ---")
    data_dir = download_and_extract_wikitext2()
    if data_dir is None:
        raise RuntimeError("无法下载或找到数据集。" )

    train_path = os.path.join(data_dir, "wiki.train.tokens")
    valid_path = os.path.join(data_dir, "wiki.valid.tokens")
    test_path = os.path.join(data_dir, "wiki.test.tokens")

    # 1. 构建词汇表
    print("正在从训练数据构建词汇表...")
    vocab = build_vocab_from_file(train_path)
    print(f"词汇表大小: {len(vocab)}")

    # 2. 将文本文件转换为 Tensors
    print("正在处理训练、验证和测试数据...")
    train_data = tokenize_file(train_path, vocab)
    valid_data = tokenize_file(valid_path, vocab)
    test_data = tokenize_file(test_path, vocab)

    # 3. 创建数据集
    train_dataset = LanguageModelDataset(train_data, bptt)
    valid_dataset = LanguageModelDataset(valid_data, bptt)
    test_dataset = LanguageModelDataset(test_data, bptt)

    # 4. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("--- 数据加载器创建完成 ---")
    return train_loader, valid_loader, test_loader, vocab

if __name__ == "__main__":
    BATCH_SIZE = 32
    BPTT = 35
    
    try:
        train_loader, valid_loader, test_loader, vocab = get_data_loaders(BATCH_SIZE, BPTT)
        
        print("\n--- 测试数据加载器 ---")
        print(f"词汇表大小: {len(vocab)}")
        
        # 从训练加载器中获取一个批次的数据
        src, tgt = next(iter(train_loader))
        
        print(f"\n单个批次的数据形状:")
        print(f"  源数据形状 (src): {src.shape}")
        print(f"  目标数据形状 (tgt): {tgt.shape}")
        
        assert src.shape == (BATCH_SIZE, BPTT), "源数据形状不正确"
        assert tgt.shape == (BATCH_SIZE, BPTT), "目标数据形状不正确"
        
        print("\n第一个样本示例:")
        print(f"  源 (tokens): {' '.join([vocab.get_word(token_id) for token_id in src[0, :15]])}")
        print(f"  目标 (tokens): {' '.join([vocab.get_word(token_id) for token_id in tgt[0, :15]])}")
        
        print("\n--- 数据加载器测试成功 ---")

    except Exception as e:
        print(f"\n--- 测试失败 ---")
        print(f"错误信息: {e}")
