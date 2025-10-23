# Transformer 模型从零实现项目

## 1. 项目简介

本项目旨在从零开始，使用 PyTorch 框架手工实现一个完整的 Transformer Encoder 模型。项目涵盖了模型核心组件的编码、训练流程的搭建，并在 WikiText-2 数据集上完成了语言建模任务的训练与评估。最终交付一个结构清晰、代码可复现的完整项目。

## 2. 环境设置

本项目依赖于 Python 3.x 和 PyTorch 框架。建议使用 Conda 或 venv 创建独立的虚拟环境。

1.  **创建并激活虚拟环境 (以 Conda 为例):**
    ```bash
    conda create -n transformer_env python=3.9
    conda activate transformer_env
    ```

2.  **安装项目依赖:**
    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 文件内容如下：
    ```
    torch
    requests
    matplotlib
    pyyaml
    ```

## 3. 运行说明

本项目提供了一个 `run.sh` 脚本，可以一键启动模型的训练和评估过程。为了确保结果的可复现性，训练过程中使用了固定的随机种子。

**执行训练:**

```bash
./scripts/run.sh
```

或者直接运行 Python 脚本：

```bash
python src/train.py
```

**超参数配置:**

所有超参数都定义在 `configs/base.yaml` 文件中，内容如下：

```yaml
model:
  embed_dim: 128
  n_heads: 4
  ffn_dim: 512
  n_layers: 2
  dropout: 0.1
training:
  batch_size: 32
  learning_rate: 3e-4
  epochs: 5
  bptt: 35
  seed: 42
```

**数据集:**

项目会自动下载并处理 WikiText-2 数据集。数据集文件将解压到项目根目录下的 `wikitext-2` 文件夹中。

## 4. 硬件要求

*   **GPU:** 建议使用 NVIDIA GPU 以加速训练过程。本项目在 NVIDIA GPU (RTX 4070) 上进行了测试。
*   **显存:** 至少 6GB 显存。
*   **CPU:** 任意现代多核 CPU。
*   **内存:** 至少 8GB RAM。

## 5. 项目结构

```
transformer-project/
├── src/                  # 存放所有核心源代码 (.py 文件)
│   ├── model.py          # Transformer模型定义
│   ├── dataset.py        # 数据集处理与加载
│   └── train.py          # 训练与评估脚本
├── configs/              # 存放配置文件
│   └── base.yaml         # 包含所有超参数的配置文件
├── scripts/
│   └── run.sh            # 一键运行训练的shell脚本
├── results/              # 存放实验结果
│   ├── training_curves.png # 训练/验证损失曲线图
│   └── metrics.json      # 最终的评估指标
├── requirements.txt      # 项目依赖库列表
└── README.md             # 项目说明文档
```
