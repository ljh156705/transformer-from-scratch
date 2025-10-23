### **项目任务书：从零实现Transformer模型并完成训练实验**

**1. 项目概述**

本项目旨在从零开始，使用 PyTorch 框架手工实现一个完整的 Transformer 模型。您需要负责模型核心组件的编码、搭建训练流程、在指定的小规模数据集上完成训练与评估，并最终交付一个结构清晰、代码可复现的完整项目。

**2. 核心任务与技术要求**

**2.1 模型实现 (从零搭建，禁止直接调用`nn.Transformer`)**

您需要独立实现 Transformer 的以下核心模块：

*   **多头自注意力机制 (Multi-Head Self-Attention):**
    *   实现可缩放的点积注意力 (Scaled Dot-Product Attention)。
    *   将多个注意力头的结果进行合并。
*   **逐位置前馈网络 (Position-wise Feed-Forward Network):**
    *   实现由两个线性层和一个激活函数构成的 FFN 模块。
*   **残差连接 (Residual Connection) 与层归一化 (Layer Normalization):**
    *   在每个子模块（自注意力和 FFN）后应用 `Add & Norm`。
*   **位置编码 (Positional Encoding):**
    *   实现基于正弦函数 (sinusoidal) 的位置编码，并将其添加到输入嵌入中。
*   **完整的 Encoder Block:**
    *   将上述模块组合成一个完整的 Transformer Encoder Block。
*   **完整的 Encoder 模型:**
    *   将多个 Encoder Block 堆叠起来，形成最终的 Encoder 模型。

**2.2 训练框架**

*   搭建完整的模型训练与验证流程。
*   实现数据加载和预处理逻辑。
*   集成优化器 (Optimizer) 和学习率调度器 (Learning Rate Scheduler)。
*   实现训练过程中的指标计算（如损失 Loss、困惑度 Perplexity）。
*   实现模型检查点 (Checkpoint) 的保存与加载功能。
*   将训练过程中的关键指标（训练/验证损失）进行可视化，并保存为图片。

**3. 实验任务与细节**

**3.1 实验任务**

*   **任务类型:** 语言建模 (Language Modeling)。
*   **模型架构:** Encoder-only Transformer。
*   **目标:** 在指定数据集上训练模型，并验证其有效性（损失需呈明显下降趋势）。

**3.2 数据集**

*   **指定数据集:** **WikiText-2**。
*   **获取方式:** 可通过 Hugging Face Datasets 平台或其它公开渠道获取。
*   **预处理要求:**
    *   对文本进行分词 (Tokenization)。
    *   构建词汇表 (Vocabulary)。
    *   将文本数据转换为模型可接受的张量格式。

**3.3 超参数设置 (必须严格遵守)**

请使用以下超参数配置进行实验：

| 参数 (Parameter)                      | 值 (Value) |
| :------------------------------------ | :--------- |
| 嵌入维度 (Embedding dimension)        | 128        |
| 注意力头数 (Number of heads)          | 4          |
| 前馈网络维度 (Feed-forward dimension) | 512        |
| Encoder层数 (Number of layers)        | 2          |
| 批次大小 (Batch size)                 | 32         |
| 学习率 (Learning rate)                | 3e-4       |
| 优化器 (Optimizer)                    | Adam       |

**3.4 评估指标**

*   主要指标: **损失 (Loss)** 和 **困惑度 (Perplexity)**。
*   您需要在每个 epoch 结束后，在验证集上计算并记录这些指标。

**4. 代码与项目结构要求**

为了确保项目的规范性和可复现性，最终交付的 GitHub 仓库必须遵循以下结构：

```
transformer-project/
├── src/                  # 存放所有核心源代码 (.py 文件)
│   ├── model.py          # Transformer模型定义
│   ├── dataset.py        # 数据集处理与加载
│   ├── train.py          # 训练与评估脚本
│   └── ...               # 其他必要的模块
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

**4.1 `README.md` 内容要求**

`README.md` 文件是项目的入口，必须包含以下内容：

1.  **项目简介:** 简要说明项目内容。
2.  **环境设置:** 清晰说明如何创建环境和安装依赖（例如 `pip install -r requirements.txt`）。
3.  **运行说明:** 提供**精确的、可直接复制粘贴的**命令行来复现整个训练过程。**必须包含随机种子**以确保结果的一致性。
    *   **示例:** `python src/train.py --config configs/base.yaml --seed 42`
4.  **硬件要求:** 说明运行实验所需的大致硬件配置（如 GPU 型号、显存大小）。

**5. 最终交付物**

您需要交付一个公开的 GitHub 仓库链接，该仓库应包含：

1.  **完整的源代码:** 遵循上述项目结构，代码需有适当的注释，清晰易读。
2.  **配置文件:** `configs/base.yaml` 文件，包含所有可调参数。
3.  **运行脚本:** `scripts/run.sh`，用于自动化执行训练流程。
4.  **依赖文件:** `requirements.txt`。
5.  **详细的 `README.md`:** 包含清晰的设置和运行指令。
6.  **实验结果:** `results/` 目录下必须包含训练曲线图和最终的指标数据。

**6. 验收标准**

1.  **完整性:** 所有在“最终交付物”中列出的内容均已提供。
2.  **可复现性:** 我将克隆您的仓库，并严格按照 `README.md` 中的指令执行。代码必须能够无误运行，并重现与您在 `results/` 目录中展示的相似的实验结果。
3.  **功能性:** 模型训练过程正常，损失曲线呈明显下降趋势，表明模型在有效学习。
4.  **规范性:** 项目结构、代码风格和文档符合本任务书的要求。

---