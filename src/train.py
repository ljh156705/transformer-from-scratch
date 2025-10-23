import torch
import torch.nn as nn
from torch.optim import Adam
import math
import time
import os
import json
import matplotlib.pyplot as plt

# 导入我们自己编写的模块
from src.dataset import get_data_loaders
from src.model import TransformerEncoder

def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch):
    """
    训练一个 epoch。
    """
    model.train() # 设置为训练模式
    total_loss = 0.
    start_time = time.time()

    for i, (src, tgt) in enumerate(data_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # 模型输出形状: [batch_size, seq_len, vocab_size]
        output = model(src)
        
        # 为了计算损失，需要将 output 和 tgt 变形
        # output -> [batch_size * seq_len, vocab_size]
        # tgt -> [batch_size * seq_len]
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()

        if i % 200 == 0 and i > 0:
            cur_loss = total_loss / (i + 1)
            elapsed = time.time() - start_time
            print(f"| epoch {epoch:3d} | {i:5d}/{len(data_loader):5d} batches | "
                  f"loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}")
            
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    """
    在验证集或测试集上评估模型。
    """
    model.eval() # 设置为评估模式
    total_loss = 0.
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def main():
    """
    主函数，执行完整的训练和评估流程。
    """
    # --- 超参数设置 (后续将从 configs/base.yaml 加载) ---
    # 模型参数
    EMBED_DIM = 128
    N_HEADS = 4
    FFN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.1
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    EPOCHS = 5 # 为了快速演示，只训练5个epoch
    BPTT = 35 # 序列长度

    # --- 环境设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据加载 ---
    # 注意：get_data_loaders 现在返回 vocab 对象
    train_loader, valid_loader, test_loader, vocab = get_data_loaders(BATCH_SIZE, BPTT)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    # --- 模型、损失函数、优化器初始化 ---
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        d_ff=FFN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 训练循环 ---
    train_losses = []
    valid_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
              f"valid loss {valid_loss:5.2f} | valid ppl {math.exp(valid_loss):8.2f}")
        print("-" * 89)

    # --- 结果保存 ---
    # 确保 results 目录存在
    os.makedirs("results", exist_ok=True)

    # 1. 保存训练曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss")
    plt.plot(range(1, EPOCHS + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plot_path = "results/training_curves.png"
    plt.savefig(plot_path)
    print(f"训练曲线图已保存到: {plot_path}")

    # 2. 保存最终评估指标
    final_metrics = {
        "final_validation_loss": valid_losses[-1],
        "final_validation_perplexity": math.exp(valid_losses[-1])
    }
    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"最终评估指标已保存到: {metrics_path}")

if __name__ == "__main__":
    main()
