# 基于多模态时序数据与知识图谱增强的脓毒症早期预警系统

## 项目概述

脓毒症是一种危及生命的严重感染并发症，如果能够提前预测，将大大提高治疗效果和患者生存率。本项目实现了一个基于人工智能的脓毒症早期预警系统，利用MIMIC-IV临床数据库中的真实患者数据，通过多模态深度学习方法预测患者发生脓毒症的风险。

### 核心功能

- **多模态数据融合**：整合生命体征、实验室检测值、药物使用记录和知识图谱信息
- **动态风险预测**：基于时序Transformer模型，为患者提供实时的脓毒症风险评分
- **知识图谱增强**：利用医学知识图谱增强临床数据，提高预测准确性
- **可视化风险轨迹**：生成患者风险随时间变化的轨迹图，辅助临床决策
- **特征重要性分析**：识别对预测结果影响最大的临床指标
- **高级数据插补**：支持多种插补方法处理缺失值，提高数据质量

## 安装指南

### 环境要求

- Python 3.8+
- CUDA 11.0+ (用于GPU加速，可选)

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/sepsis-early-warning.git
cd sepsis-early-warning
```

2. 创建虚拟环境（可选）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

1. 获取MIMIC-IV数据库访问权限（需通过PhysioNet的访问申请）
2. 处理后的数据存放在`data/processed/`目录下，包括：
   - 患者基本信息 (`all_patients.csv`)
   - 生命体征数据 (`vital_signs_interim_5.csv`)
   - 实验室检测值 (`lab_values.csv`)
   - 时间序列数据 (`time_series.csv`)

### 数据处理与插补

处理缺失数据是临床分析的关键步骤。本项目提供多种插补方法：

1. 使用简单插补（均值、中位数、众数填充）

```bash
python src/data_processor/run_imputation.py --method simple --strategy mean
```

2. 使用K近邻（KNN）插补

```bash
python src/data_processor/run_imputation.py --method knn --n_neighbors 5
```

3. 使用多重插补链方程（MICE）

```bash
python src/data_processor/run_imputation.py --method mice
```

4. 使用时间序列前向填充

```bash
python src/data_processor/run_imputation.py --method forward_fill
```

5. 特征选择（移除高缺失率特征）

```bash
python src/data_processor/run_imputation.py --method feature_selection --missing_threshold 0.5
```

### 插补方法选择指南

不同类型的数据适合不同的插补方法：

- **生命体征数据**：时间连续性强，建议使用前向填充(`forward_fill`)
- **实验室检测结果**：间隔较大，建议使用KNN或MICE插补
- **高缺失率特征**：当缺失率超过50%，建议通过特征选择移除
- **混合策略**：可以在主程序中使用前向填充，对时间序列数据进行实时处理

### 运行系统

1. 训练模型

```bash
python src/main.py --data_dir ./data/processed --output_dir ./output
```

2. 查看结果

训练完成后，可以在`output/run_YYYYMMDD_HHMMSS/`目录下查看结果：
- `sepsis_prediction_results.html`：评估报告
- `figures/`：包含ROC曲线、混淆矩阵、风险轨迹图和特征重要性图

### 配置参数

可以通过命令行参数调整系统配置：

```bash
python src/main.py --data_dir ./data/processed --max_samples 3000 --hidden_dim 128 --num_heads 4 --batch_size 32 --epochs 50 --imputation_method forward_fill
```

主要参数说明：
- `--data_dir`：数据目录路径
- `--max_samples`：最大样本数量
- `--hidden_dim`：模型隐藏层维度
- `--num_heads`：Transformer注意力头数量
- `--num_layers`：Transformer编码器层数
- `--batch_size`：批处理大小
- `--epochs`：训练轮数
- `--kg_method`：知识图谱嵌入方法（TransE/DistMult/ComplEx）
- `--imputation_method`：缺失值插补方法（simple/knn/mice/forward_fill）

## 系统架构

### 模块结构

- **数据处理模块** (`src/data_processor/`)：处理和预处理多模态医疗数据
  - `data_loader.py`：加载和分割数据
  - `data_imputer.py`：实现多种缺失值插补方法
  - `run_imputation.py`：独立运行的插补脚本
- **知识图谱模块** (`src/knowledge_graph/`)：构建医学知识图谱和生成实体嵌入
- **模型模块** (`src/models/`)：定义多模态Transformer模型
- **训练模块** (`src/training/`)：实现模型训练循环和早停机制
- **评估模块** (`src/evaluation/`)：评估模型性能和生成评估指标
- **可视化模块** (`src/visualization/`)：生成ROC曲线、混淆矩阵、风险轨迹图等

### 技术细节

- **多模态融合**：通过特征投影层将不同类型数据映射到统一维度空间
- **时间感知**：使用位置编码和时间嵌入层捕捉时序信息
- **知识图谱嵌入**：采用TransE方法将医学概念映射到低维稠密向量
- **动态预测**：基于Transformer架构实现对长序列时序数据的建模
- **可解释性分析**：通过特征重要性和注意力权重分析提供模型解释
- **高级数据插补**：提供多种插补算法处理临床数据中的缺失值

## 引用

如果您在研究中使用了本系统，请引用以下文献：

- Sepsis-3共识：Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810.
- MIMIC-IV数据库：Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023.

## 联系方式

如有问题或建议，请联系：

- 项目维护者：您的姓名
- 电子邮件：your.email@example.com
- GitHub Issues：https://github.com/yourusername/sepsis-early-warning/issues 