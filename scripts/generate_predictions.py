#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成脓毒症预测结果并保存到CSV文件
"""

import os
import torch
import pandas as pd
import numpy as np
from utils.data_loading import load_structured_data, preprocess_features
from utils.dataset import SepsisDataset
from models.fusion.multimodal_model import SepsisTransformerModel
from config import FEATURE_CONFIG, MODEL_CONFIG

def generate_predictions():
    """生成模型预测结果并保存到CSV文件"""
    print("生成预测结果文件...")
    
    # 1. 加载数据
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 2. 加载模型
    print("加载训练好的模型...")
    model = SepsisTransformerModel(
        vitals_dim=MODEL_CONFIG['vitals_dim'],
        lab_dim=MODEL_CONFIG['lab_dim'],
        drug_dim=MODEL_CONFIG['drug_dim'],
        text_dim=MODEL_CONFIG['text_dim'],
        kg_dim=MODEL_CONFIG['kg_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    # Windows路径兼容
    model_path = 'models\\best_model.pt'
    if not os.path.exists(model_path):
        model_path = 'models/best_model.pt'  # 尝试使用Linux风格路径
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. 加载测试集索引
    splits_path = 'models\\dataset_splits.pt'
    if not os.path.exists(splits_path):
        splits_path = 'models/dataset_splits.pt'
    
    try:
        dataset_splits = torch.load(splits_path)
        test_indices = dataset_splits['test_indices']
        print(f"已加载测试集索引，共{len(test_indices)}个样本")
    except Exception as e:
        print(f"加载测试集索引失败: {e}")
        print("错误：无法继续生成预测，请确保已完成模型训练")
        return
    
    # 生成实际预测结果
    print("基于测试集生成预测结果...")

    # 预处理数据，与analysis.py中相同
    vitals, labs, drugs, text_embeds = preprocess_features(patient_features, FEATURE_CONFIG)

    # 参数设置
    max_seq_len = 72  # 默认最大序列长度
    num_patients = len(patient_features['subject_id'].unique())
    print(f"总数据量: {len(patient_features)}, 患者数量: {num_patients}")

    # 获取唯一的患者ID
    patient_ids = patient_features['subject_id'].unique()
    patient_id_map = {pid: str(pid) for pid in patient_ids}  # 为lookup创建映射

    # 处理数据维度
    from utils.dataset import SepsisDataset
    # ... 处理代码与analysis.py中相同 ...

    # 从测试集索引中选择患者
    selected_test_indices = test_indices[:100]  # 直接使用列表切片，移除.numpy()调用
    predictions = []

    # 批量进行预测，以提高效率
    batch_size = 16
    for i in range(0, len(selected_test_indices), batch_size):
        # 获取当前批次的索引
        batch_indices = selected_test_indices[i:i+batch_size]
        
        # 准备批次数据
        batch_vitals = []
        batch_labs = []
        batch_drugs = []
        batch_text_embeds = []
        batch_kg_embeds = []
        batch_time_indices = []
        batch_patient_ids = []
        
        for idx in batch_indices:
            if idx >= len(patient_ids):
                continue  # 跳过无效索引
            
            patient_id = patient_ids[idx]
            batch_patient_ids.append(str(patient_id))
            
            # 获取患者数据
            patient_data_indices = patient_features['subject_id'] == patient_id
            
            # 获取特征
            patient_vitals = vitals[patient_data_indices]
            patient_labs = labs[patient_data_indices]
            patient_drugs = drugs[patient_data_indices]
            patient_text_embeds = text_embeds[patient_data_indices]
            
            # 获取时间信息和真实标签
            sepsis_data_indices = sepsis_labels['subject_id'] == patient_id
            if 'hour' in sepsis_labels.columns:
                patient_times = sepsis_labels.loc[sepsis_data_indices, 'hour'].values
            else:
                patient_times = np.arange(len(patient_vitals))
                
            # 获取真实标签
            if 'sepsis_label' in sepsis_labels.columns:
                patient_actual_labels = sepsis_labels.loc[sepsis_data_indices, 'sepsis_label'].values
            else:
                patient_actual_labels = np.zeros(len(patient_times))
            
            seq_len = min(len(patient_times), max_seq_len, len(patient_vitals))
            if seq_len == 0:
                continue  # 跳过没有数据的患者
            
            # 处理特征
            reshaped_vitals = np.zeros((max_seq_len, vitals.shape[1]))
            reshaped_labs = np.zeros((max_seq_len, labs.shape[1]))
            reshaped_drugs = np.zeros((max_seq_len, drugs.shape[1]))
            reshaped_time_indices = np.zeros(max_seq_len)
            
            # 填充数据
            reshaped_vitals[:seq_len] = patient_vitals[:seq_len]
            reshaped_labs[:seq_len] = patient_labs[:seq_len]
            reshaped_drugs[:seq_len] = patient_drugs[:seq_len]
            reshaped_time_indices[:seq_len] = np.arange(1, seq_len + 1)
            
            # 获取文本嵌入 (取平均)
            avg_text_embed = np.mean(patient_text_embeds, axis=0) if len(patient_text_embeds) > 0 else np.zeros(text_embeds.shape[1])
            
            # 扩展文本嵌入为3D
            text_embed_expanded = np.zeros((max_seq_len, text_embeds.shape[1]))
            for t in range(max_seq_len):
                text_embed_expanded[t] = avg_text_embed
            
            # 获取知识图谱嵌入 (使用第一个)
            kg_embed = kg_embeddings[0] if len(kg_embeddings) > 0 else np.zeros(kg_embeddings.shape[1])
            
            # 添加到批次
            batch_vitals.append(reshaped_vitals)
            batch_labs.append(reshaped_labs)
            batch_drugs.append(reshaped_drugs)
            batch_text_embeds.append(text_embed_expanded)
            batch_kg_embeds.append(kg_embed)
            batch_time_indices.append(reshaped_time_indices)
        
        if not batch_vitals:
            continue  # 跳过空批次
        
        # 转换为PyTorch张量
        batch_vitals = torch.tensor(np.array(batch_vitals), dtype=torch.float32).to(device)
        batch_labs = torch.tensor(np.array(batch_labs), dtype=torch.float32).to(device)
        batch_drugs = torch.tensor(np.array(batch_drugs), dtype=torch.float32).to(device)
        batch_text_embeds = torch.tensor(np.array(batch_text_embeds), dtype=torch.float32).to(device)
        batch_kg_embeds = torch.tensor(np.array(batch_kg_embeds), dtype=torch.float32).to(device)
        batch_time_indices = torch.tensor(np.array(batch_time_indices), dtype=torch.long).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(batch_vitals, batch_labs, batch_drugs, batch_text_embeds, batch_kg_embeds, batch_time_indices)
            risk_scores = outputs.cpu().numpy()
        
        # 计算特征重要性
        for b, patient_id in enumerate(batch_patient_ids):
            for h in range(12):  # 取前12小时
                if h >= risk_scores.shape[1]:
                    continue  # 跳过超过序列长度的时间点
                
                # 风险评分
                risk_score = float(risk_scores[b, h, 0]) if risk_scores[b, h, 0] <= 1.0 else 1.0
                risk_score = max(0.0, min(1.0, risk_score))  # 确保在[0,1]范围内
                
                # 获取实际标签 - 使用真实标签
                actual_label = 0
                patient_data_indices = patient_features['subject_id'] == int(patient_id)
                sepsis_data_indices = sepsis_labels['subject_id'] == int(patient_id)
                if len(sepsis_data_indices) > 0 and h < len(sepsis_labels.loc[sepsis_data_indices, 'sepsis_label'].values):
                    actual_label = int(sepsis_labels.loc[sepsis_data_indices, 'sepsis_label'].values[h])
                
                # 计算特征重要性
                # TODO: 实现基于SHAP或特征归因的真实特征重要性计算
                # 目前仅使用基本特征作为占位符
                feature_importances = {
                    '心率': 0.0,
                    '呼吸频率': 0.0,
                    '收缩压': 0.0,
                    'WBC': 0.0,
                    '乳酸': 0.0
                }
                
                # 从模型结果中提取真实重要性 - 这一部分需要根据您的模型架构实现
                # 这里暂时使用相等的权重
                for feature in feature_importances:
                    feature_importances[feature] = 0.2  # 平均分配权重
                
                # 添加到预测结果
                predictions.append({
                    'patient_id': patient_id,
                    'hour': h,
                    'predicted_risk': risk_score,
                    'actual_label': actual_label,
                    'feature_importance': ','.join([f"{k}:{v:.4f}" for k, v in feature_importances.items()])
                })
        
        print(f"已生成批次预测: {len(batch_patient_ids)} 患者")

    # 保存预测结果
    df = pd.DataFrame(predictions)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results\\predictions.csv', index=False)
    print(f"已生成预测结果并保存到results\\predictions.csv，共{len(predictions)}条记录")

if __name__ == "__main__":
    generate_predictions() 