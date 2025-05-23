# 基于MIMIC-IV数据的脓毒症早期预警系统

## 项目概述

本项目实现了一个基于MIMIC-IV数据库的脓毒症早期预警系统，通过分析患者的多源时序数据，预测脓毒症发生风险，并提供可解释的预警结果。系统利用真实的ICU数据，构建了完整的数据处理、模型训练和结果可视化流程。

### 核心功能

1. **多维数据整合**：处理MIMIC-IV数据库中的生命体征、实验室检查、药物使用等多维时序数据
2. **脓毒症预测**：基于Sepsis-3标准构建脓毒症早期预警模型
3. **风险评估**：动态计算患者发展为脓毒症的风险概率
4. **可视化分析**：提供ROC曲线、风险轨迹等多种直观的可视化结果

### 技术架构

- **数据处理层**：PostgreSQL数据库连接、SQL查询优化、时序数据处理
- **模型层**：多模态Transformer模型，整合生命体征、实验室检测等多源数据
- **预测层**：动态风险评估、特征重要性分析
- **可视化层**：交互式HTML报告、图表展示

## 环境要求

- Python 3.9+
- PostgreSQL 12+
- MIMIC-IV数据库
- 至少8GB RAM
- 至少20GB磁盘空间
- CUDA支持（推荐，但非必需）

## 快速安装

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

## 数据库配置

系统需要访问MIMIC-IV数据库。需要设置以下环境变量以配置数据库连接：

```bash
# Windows PowerShell
$env:MIMIC_DB_HOST = "172.16.3.67"
$env:MIMIC_DB_PORT = "5432"
$env:MIMIC_DB_NAME = "mimiciv"
$env:MIMIC_DB_USER = "postgres"
$env:MIMIC_DB_PASSWORD = "123456"

# 或者使用提供的脚本
.\set_mimic_env.ps1
```

## 执行流程

### 1. 测试数据库连接

首先测试数据库连接是否正常：

```bash
python scripts/test_db_connection.py
```

### 2. 提取数据

从MIMIC-IV数据库中提取脓毒症相关数据：

```bash
python scripts/fixed_extract_sepsis_data.py
```

### 3. 处理数据

处理提取的数据，创建用于模型训练的数据集：

```bash
# 处理3000名患者的数据
python scripts/process_sepsis_data.py --sample_patients 3000

# 如需处理较小样本或遇到内存问题，可以减少样本量
python scripts/process_sepsis_data.py --sample_patients 1000
```

### 4. 转换数据格式

将处理后的数据转换为模型所需的格式：

```bash
python scripts/convert_processed_data.py
```

### 5. 训练模型

训练脓毒症预测模型：

```bash
python scripts/model_training.py
```

### 6. 生成预测结果

使用训练好的模型生成预测结果：

```bash
python scripts/generate_predictions.py
```

### 7. 可视化分析

生成可视化结果和解释性分析：

```bash
python utils/explanation.py
```

### 8. 运行完整系统

运行完整的脓毒症早期预警系统（跳过已完成的数据提取步骤）：

```bash
python run_complete_sepsis_system.py --skip_extraction
```

或者只运行可视化部分：

```bash
python run_complete_sepsis_system.py --only_viz
```

### 9. 查看结果

系统生成的结果可在 `results` 目录下查看：

- `results/sepsis_prediction_results.html` - 交互式预测结果报告
- `results/figures/` - 包含各类可视化图表
- `results/evaluation_results.json` - 模型评估结果

## 系统输出

系统将生成以下输出结果：

- `results/sepsis_prediction_results.html` - 交互式预测结果报告
- `results/figures/roc_curve.png` - ROC曲线图
- `results/figures/confusion_matrix.png` - 混淆矩阵
- `results/figures/risk_trajectories.png` - 风险轨迹图
- `results/evaluation_results.json` - 包含精确度、召回率、F1分数等指标的评估结果

## 项目结构

```
├── data/                 # 数据目录
│   ├── processed/        # 处理后的数据
│   └── knowledge_graph/  # 知识图谱数据
├── models/               # 模型目录
├── scripts/              # 脚本目录
│   ├── fixed_extract_sepsis_data.py  # 数据提取
│   ├── process_sepsis_data.py        # 数据处理
│   ├── convert_processed_data.py     # 数据格式转换
│   ├── test_db_connection.py         # 数据库连接测试
│   └── model_training.py             # 模型训练
├── utils/                # 工具函数
│   ├── database_config.py     # 数据库配置
│   ├── data_loading.py        # 数据加载
│   ├── evaluation.py          # 评估指标
│   └── explanation.py         # 特征解释
├── results/              # 结果输出目录
├── config.py             # 配置文件
├── run_complete_sepsis_system.py  # 完整系统执行脚本
└── README.md             # 项目说明
```

## 故障排除

1. **数据库连接问题**：
   - 确保PostgreSQL服务正在运行
   - 验证数据库连接参数是否正确
   - 使用`scripts/test_db_connection.py`测试连接

2. **编码问题**：
   - 如果遇到UTF-8编码错误，请使用文本编辑器打开文件，另存为UTF-8格式（不带BOM）
   - 对于Windows用户，可能需要处理文件中的null字节

3. **内存问题**：
   - 如果遇到内存错误，尝试减少处理的患者数量：`--sample_patients 500`
   - 关闭其他内存密集型应用程序

## 参考资料

- MIMIC-IV数据集: https://physionet.org/content/mimiciv/
- Sepsis-3定义: Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA 2016 