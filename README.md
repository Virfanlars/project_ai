# 基于多模态时序数据与知识图谱增强的脓毒症早期预警系统

## 项目概述

本项目实现了一个基于多模态时序数据和知识图谱增强的脓毒症早期预警系统，旨在通过分析患者的多源异构数据，尽早识别脓毒症发生风险，并提供可解释的预警结果。

### 技术亮点

1. **多模态数据融合**：整合MIMIC数据集中的生命体征（心率、呼吸频率等）、实验室数据（白细胞计数、乳酸值等）、用药记录（抗生素使用时间）以及护理记录中的自由文本（如"感染疑似描述"），通过Transformer模型进行时序特征提取与跨模态对齐。

2. **双时序建模架构**：同时实现LSTM和TCN两种时序建模方法，LSTM更擅长捕捉长依赖关系，TCN更有效处理平行特征，双模型互补提升预测效果。

3. **知识图谱嵌入**：构建医学知识图谱，将ICD诊断编码、药品相互作用等结构化知识融入模型，增强特征的可解释性。

4. **动态风险预测**：利用时间注意力机制设计动态预警模型，结合SHAP值实时输出风险概率及关键影响因素。

5. **时序SHAP分析**：创新性地实现了时序SHAP分析方法，可视化特征重要性随时间动态变化过程，提升临床可解释性。

6. **大规模数据集**：使用完整MIMIC-IV数据集，通过数据增强技术显著扩大训练样本，提高模型泛化能力。

## 系统架构

系统架构采用多级流水线设计，包含数据处理、特征工程、模型训练和解释几个主要模块：

1. **数据处理模块**：连接MIMIC-IV数据库，提取患者生命体征、实验室检查和用药记录等多模态数据。
2. **特征工程模块**：对时序数据进行处理，构建知识图谱，生成文本嵌入。
3. **模型训练模块**：训练融合多模态数据的LSTM和TCN预测模型。
4. **解释性模块**：使用SHAP分析特征重要性，生成动态风险评估结果。

## 环境配置

```bash
# 创建虚拟环境
python -m venv sepsis_env

# 激活环境
# Windows:
sepsis_env\Scripts\activate
# Linux/Mac:
source sepsis_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

本项目使用MIMIC-IV数据集，需要先申请访问权限并按照以下步骤处理数据：

1. 下载MIMIC-IV数据集并导入本地PostgreSQL数据库
2. 配置数据库连接信息 (在`utils/database_config.py`中修改)
3. 运行数据抽取脚本处理原始数据

## 项目结构

```
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 预处理后的数据
│   └── knowledge_graph/  # 知识图谱数据
├── models/               # 模型目录
│   ├── fusion/           # 多模态融合模型
│   ├── knowledge_graph/  # 知识图谱模型
│   ├── text/             # 文本处理模型
│   └── time_series/      # 时序预测模型(LSTM和TCN)
├── scripts/              # 脚本目录
│   ├── data_extraction_main.py  # 数据抽取
│   ├── model_training.py        # 模型训练与评估
│   ├── analysis.py              # 模型解释与分析
│   └── sepsis_labeling.py       # 脓毒症标签构建
├── utils/                # 工具函数
│   ├── database_config.py     # 数据库配置
│   ├── evaluation.py          # 评估指标
│   ├── visualization.py       # 可视化工具
│   └── explanation.py         # SHAP分析工具
├── results/              # 结果输出目录
├── config.py             # 配置文件
├── run_project.py        # 主执行脚本
├── run_sepsis_system.py  # 完整系统执行脚本
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 执行流程

```bash
# 运行完整项目流程
python run_project.py

# 运行完整系统（包含数据处理、模型训练和可视化）
python run_sepsis_system.py

# 仅运行系统中的可视化部分
python run_sepsis_system.py --only_viz

# 使用样本数据运行系统
python run_sepsis_system.py --sample

# 仅执行训练
python train.py

# 仅执行预测
python predict.py --patient_id <患者ID>
```

## 技术实现细节

### 1. 多模态数据融合

- 使用Transformer架构进行多模态时序数据融合
- 对不同模态的数据进行特征投影，统一到相同的特征空间
- 添加位置编码以保留时序信息

### 2. 双时序模型架构

- LSTM模型：使用多层双向LSTM结合注意力机制，捕捉长期依赖关系
- TCN模型：采用扩张卷积和残差连接，有效处理变长时序数据
- 集成策略：使用模型集成方法，结合两种模型的预测优势

### 3. 知识图谱嵌入

- 构建包含疾病、药物、实验室检查的医学知识图谱
- 使用关系图卷积网络(RGCN)生成知识嵌入
- 将知识嵌入融入患者特征表示

### 4. SHAP可解释性分析

- 全局特征重要性分析：识别对模型预测影响最大的特征
- 患者个体SHAP分析：解释针对特定患者的预测依据
- 时序SHAP分析：追踪特征重要性随时间变化的模式

## 参考资料

- MIMIC-IV数据集: https://physionet.org/content/mimiciv/
- Transformer模型: Vaswani et al., "Attention is All You Need," NIPS 2017
- 知识图谱嵌入: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks," ESWC 2018
- LSTM: Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation 1997
- TCN: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," 2018
- SHAP值: Lundberg et al., "A Unified Approach to Interpreting Model Predictions," NIPS 2017

# 脓毒症早期预警系统

基于MIMIC-IV数据库的多模态脓毒症早期预警系统，整合生理指标、实验室检查、药物使用和临床文本数据，实时预测患者发展为脓毒症的风险。

## 系统环境要求

- Python 3.9+
- PostgreSQL 12+
- MIMIC-IV数据库
- 至少8GB RAM
- NVIDIA GPU (推荐)

## 安装与配置

1. 克隆代码库
2. 安装依赖包：`pip install -r requirements.txt`
3. 配置数据库连接环境变量（使用提供的脚本）

## 运行系统

### 方法1：使用CMD脚本（推荐）

双击 `run_mimic.cmd` 文件，然后按照屏幕提示选择运行模式：

1. 仅运行可视化
2. 运行系统（跳过数据库操作）
3. 运行完整系统

### 方法2：手动设置环境变量并运行

#### 在CMD命令行中：

```
.\set_mimic_env_fixed.bat
python run_complete_sepsis_system.py --only_viz
```

#### 在PowerShell中：

首先需要允许运行未签名脚本（仅在当前会话中有效）：

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\set_mimic_env.ps1
```

### 运行模式

- **仅可视化模式**：`--only_viz`
  生成模拟数据的可视化结果，不需要数据库连接
  
- **跳过数据库模式**：`--skip_db`
  使用本地缓存的数据运行系统，不需要连接MIMIC数据库
  
- **完整模式**：`--force_real_data`
  运行完整系统，包括数据库连接、数据提取、模型训练和可视化

## 输出结果

系统将生成以下输出：

- `results/sepsis_prediction_results.html` - 主要报告
- `results/figures/` - 包含ROC曲线、风险轨迹等多种可视化图表

## 故障排除

1. **编码问题**：如果遇到字符编码错误（如null字节错误），可以使用记事本打开相关文件，保存为UTF-8格式（不带BOM）。

2. **PowerShell执行策略错误**：使用以下命令临时允许运行脚本：
   ```
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. **数据库连接错误**：检查是否正确设置了环境变量，确保数据库服务器地址和端口正确。 