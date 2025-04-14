# 模型和训练配置
import torch

# 模型配置
MODEL_CONFIG = {
    'vitals_dim': 6,  # 生命体征特征维度
    'lab_dim': 5,     # 实验室检查特征维度
    'drug_dim': 9,    # 药物特征维度 (5种抗生素 + 4种血管活性药物)
    'text_dim': 768,  # 文本嵌入维度 (Bio_ClinicalBERT输出)
    'kg_dim': 64,     # 知识图谱嵌入维度
    'hidden_dim': 512, # 大幅增加隐藏层维度，提高模型容量
    'num_heads': 16,   # 显著增加注意力头数
    'num_layers': 8,  # 大幅增加层数，增强模型深度
    'dropout': 0.2,   # 保持适当的dropout防止过拟合
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 128,  # 大幅增加批量大小，充分利用GPU
    'num_epochs': 500, # 显著增加训练轮数，确保充分训练
    'learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'early_stopping_patience': 30, # 增加早停耐心，避免过早停止
    'lr_scheduler': True,  # 启用学习率调度器
    'lr_scheduler_patience': 15,
    'lr_scheduler_factor': 0.5,
    'gradient_clip': 1.0,  # 添加梯度裁剪，防止梯度爆炸
    'use_mixed_precision': True, # 启用混合精度训练，加速训练过程
}

# 特征配置
FEATURE_CONFIG = {
    'vitals_columns': ['heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2'],
    'labs_columns': ['wbc', 'lactate', 'creatinine', 'platelet', 'bilirubin'],
    'drugs_columns': ['antibiotic_1', 'antibiotic_2', 'antibiotic_3', 'antibiotic_4', 'antibiotic_5',
                     'vasopressor_1', 'vasopressor_2', 'vasopressor_3', 'vasopressor_4'],
    'text_embed_columns': [f'text_embed_{i}' for i in range(768)],  # BERT嵌入特征列
}

# 数据处理配置
DATA_CONFIG = {
    'time_resolution': '1H',  # 时间分辨率
    'max_seq_len': 72,        # 增加最大序列长度 (3天)
    'train_ratio': 0.7,       # 训练集比例
    'val_ratio': 0.15,        # 验证集比例
    'test_ratio': 0.15,       # 测试集比例
    'use_upsampling': True,   # 使用上采样处理不平衡数据
    'use_data_augmentation': True,  # 使用数据增强
    'min_patient_count': 10000, # 设定最小患者数量
    'use_all_mimic_data': True, # 使用全部MIMIC数据
    'time_shift_augmentation': True, # 启用时间偏移数据增强
    'value_perturbation': True, # 启用数值扰动增强
} 