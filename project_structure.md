# 脓毒症早期预警系统 - 项目目录结构

```
/
├── data/                      # 数据处理模块
│   ├── extraction/            # 数据提取脚本
│   ├── preprocessing/         # 数据预处理脚本
│   └── integration/           # 多模态数据集成
├── models/                    # 模型实现
│   ├── text/                  # 文本处理模型
│   ├── vitals/                # 生命体征模型
│   ├── labs/                  # 实验室检测模型
│   ├── fusion/                # 多模态融合模型
│   └── knowledge/             # 知识图谱嵌入
├── scripts/                   # 执行脚本
│   ├── generate_derived_tables.py  # 派生表生成
│   └── train_model.py         # 模型训练脚本
├── utils/                     # 工具函数
│   ├── database_config.py     # 数据库配置
│   ├── evaluation.py          # 评估指标
│   ├── visualization.py       # 可视化工具
│   └── explanation.py         # 解释性分析
├── notebooks/                 # Jupyter笔记本
│   ├── exploratory_analysis.ipynb    # 探索性分析
│   └── model_evaluation.ipynb        # 模型评估
├── config/                    # 配置文件
│   ├── model_config.yaml      # 模型配置
│   └── feature_config.yaml    # 特征配置
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
``` 