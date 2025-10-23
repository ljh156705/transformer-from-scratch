

## 1. PROJECT CHARTER  
      
**项目执行计划**

我将项目分解为以下 8 个核心步骤，并会依次执行：

1.  **步骤一：项目结构初始化**
    *   **任务描述:** 根据您的要求，创建完整的项目目录结构，包括 `src/`, `configs/`, `scripts/`, 和 `results/`。
    *   **产出物:** 空的项目文件夹结构。

2.  **步骤二：环境与配置**
    *   **任务描述:** 创建 `requirements.txt` 文件并列出所有必要的依赖库（如 `torch`, `datasets`, `transformers` 等）。同时，创建 `configs/base.yaml` 文件，并填入您指定的超参数。
    *   **产出物:** `requirements.txt` 和 `configs/base.yaml` 文件。

3.  **步骤三：数据处理模块 (`src/dataset.py`)**
    *   **任务描述:** 编码实现数据加载和预处理逻辑。这包括从 Hugging Face Hub 下载 WikiText-2 数据集，进行分词，构建词汇表，并将其转换为 PyTorch Tensors。
    *   **产出物:** `src/dataset.py` 脚本文件。

4.  **步骤四：模型核心实现 (`src/model.py`)**
    *   **任务描述:** 从零开始，手工编写 Transformer Encoder 模型的全部核心组件，包括：可缩放点积注意力、多头注意力机制、前馈网络、位置编码以及完整的 Encoder Block。**绝不直接调用 `nn.Transformer`**。
    *   **产出物:** `src/model.py` 脚本文件。

5.  **步骤五：训练与评估脚本 (`src/train.py`)**
    *   **任务描述:** 编写主训练脚本。该脚本将负责：
        *   加载配置文件和数据集。
        *   初始化模型、优化器和学习率调度器。
        *   执行完整的训练和验证循环。
        *   计算损失（Loss）和困惑度（Perplexity）。
        *   在训练结束后，将训练/验证损失曲线图保存到 `results/training_curves.png`。
        *   将最终的评估指标保存到 `results/metrics.json`。
    *   **产出物:** `src/train.py` 脚本文件。

6.  **步骤六：一键运行脚本 (`scripts/run.sh`)**
    *   **任务描述:** 创建一个简单的 shell 脚本，用于封装训练命令，实现一键复现。
    *   **产出物:** `scripts/run.sh` 文件。

7.  **步骤七：项目文档 (`README.md`)**
    *   **任务描述:** 撰写详细的 `README.md` 文件，内容将严格按照您的要求，包括项目简介、环境设置、精确的运行指令（包含随机种子）和硬件要求。
    *   **产出物:** `README.md` 文件。

8.  **步骤八：完整流程执行与验证**
    *   **任务描述:** 执行 `scripts/run.sh`，完整运行一次训练流程，确保所有代码正常工作，并生成最终的实验结果文件。
    *   **产出物:** 填充了结果的 `results/` 目录，以及一个经过验证的可复现项目。  
 
## 2. TASK DECOMPOSITION CHECKLIST

- [x] **步骤一：项目结构初始化**
- [x] **步骤二：环境与配置 (已创建 `requirements.txt` 和 `configs/base.yaml`)**
- [x] **步骤三：数据处理模块 (`dataset.py`)**
- [x] **步骤四：模型核心实现 (`model.py`)**
- [x] **步骤五：训练与评估脚本 (`train.py`)**
- [x] **步骤六：一键运行脚本 (`scripts/run.sh`)**
- [x] **步骤七：项目文档 (`README.md`)**
- [x] **步骤八：完整流程执行与验证**   

## 3. LAST UPDATED TIMESTAMP

2025-10-23 18:01:35
