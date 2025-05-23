# 模型和训练配置
import torch

# 模型配置
MODEL_CONFIG = {
    'vitals_dim': 6,  # 生命体征特征维度
    'lab_dim': 5,     # 实验室检查特征维度
    'drug_dim': 9,    # 药物特征维度 (5种抗生素 + 4种血管活性药物)
    'text_dim': 32,   # 降低文本嵌入维度，简化模型
    'kg_dim': 64,     # 知识图谱嵌入维度
    'hidden_dim': 64, # 减小隐藏层维度，降低复杂度
    'num_heads': 2,   # 减少注意力头数
    'num_layers': 1,  # 减少Transformer层数
    'dropout': 0.2,   # 保持适当的dropout防止过拟合
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 64,   # 减小批量大小
    'num_epochs': 50,  # 减少训练轮数，先确认模型能够学习
    'learning_rate': 0.001,  # 增大学习率加快收敛
    'weight_decay': 1e-4,    # 增加权重衰减防止过拟合
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'early_stopping_patience': 10,  # 减少早停耐心
    'lr_scheduler': True,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'gradient_clip': 1.0,
    'use_mixed_precision': False,  # 关闭混合精度，简化训练
    'class_weights': [0.4, 0.6],   # 添加类别权重，处理不平衡
}

# 特征配置
FEATURE_CONFIG = {
    'vitals_columns': ['heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2'],
    'labs_columns': ['wbc', 'lactate', 'creatinine', 'platelet', 'bilirubin'],
    'drugs_columns': ['antibiotic_1', 'antibiotic_2', 'antibiotic_3', 'antibiotic_4', 'antibiotic_5',
                     'vasopressor_1', 'vasopressor_2', 'vasopressor_3', 'vasopressor_4'],
    'text_embed_columns': [f'text_embed_{i}' for i in range(32)],  # 减少文本嵌入特征数量
    'use_mean_imputation': True,   # 使用均值填充代替零填充
    'use_standardization': True,   # 使用标准化处理特征
    'use_forward_fill': True,      # 使用前向填充处理时序缺失值
}

# 数据处理配置
DATA_CONFIG = {
    'time_resolution': '1H',
    'max_seq_len': 48,       # 减小最大序列长度
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'use_upsampling': True,   # 使用上采样处理不平衡
    'use_data_augmentation': False,  # 关闭数据增强简化调试
    'min_patient_count': 500,  # 减少患者数量加快训练
    'use_all_mimic_data': False, # 先不使用全部数据
    'time_shift_augmentation': False,
    'value_perturbation': False,
    'target_column': 'sepsis_label',  # 明确指定使用sepsis_label作为目标列
} 