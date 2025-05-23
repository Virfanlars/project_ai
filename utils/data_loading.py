import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def load_structured_data(use_sample_data=False):
    """
    加载处理好的患者特征、标签和知识图谱嵌入数据
    
    参数:
        use_sample_data: 是否使用示例数据
        
    返回:
        patient_features: 患者特征DataFrame
        sepsis_labels: 脓毒症标签DataFrame
        kg_embeddings: 知识图谱嵌入numpy数组
        time_axis: 时间轴信息
    """
    # 如果存在处理好的数据，则直接加载
    if os.path.exists('data/processed/patient_features.csv') and not use_sample_data:
        print("从本地文件加载数据...")
        
        # 加载患者特征
        patient_features = pd.read_csv('data/processed/patient_features.csv')
        
        # 加载脓毒症标签
        sepsis_labels = pd.read_csv('data/processed/sepsis_labels.csv')
        
        # 加载知识图谱嵌入
        kg_embeddings = np.load('data/processed/kg_embeddings.npy')
        
        # 加载时间轴信息
        with open('data/processed/time_axis.json', 'r') as f:
            time_axis = json.load(f)
    else:
        print("错误：找不到必要的数据文件")
        print("请确保数据处理步骤已完成，并且数据文件存在于 data/processed/ 目录")
        return None, None, None, None
    
    return patient_features, sepsis_labels, kg_embeddings, time_axis


# def generate_sample_data(n_patients=1000, n_hours=48, embedding_dim=64):
#     """
#     生成示例数据用于开发和测试
    
#     参数:
#         n_patients: 患者数量
#         n_hours: 每个患者的小时数
#         embedding_dim: 文本嵌入维度
        
#     返回:
#         patient_features: 患者特征DataFrame
#         sepsis_labels: 脓毒症标签DataFrame
#         kg_embeddings: 知识图谱嵌入
#         time_axis: 时间轴信息
#     """
#     print(f"生成{n_patients}名患者的示例数据，每名患者{n_hours}小时...")
    
#     # 生成患者ID和住院ID
#     subject_ids = list(range(1000, 1000 + n_patients))
#     hadm_ids = list(range(2000, 2000 + n_patients))
#     stay_ids = list(range(3000, 3000 + n_patients))
    
#     # 生成时间轴
#     reference_time = datetime(2022, 1, 1)
#     hours = list(range(n_hours))
    
#     # 生成特征数据
#     data = []
#     for i, (subject_id, hadm_id, stay_id) in enumerate(zip(subject_ids, hadm_ids, stay_ids)):
#         for hour in hours:
#             # 创建随机生命体征和实验室值
#             row = {
#                 'subject_id': subject_id,
#                 'hadm_id': hadm_id,
#                 'stay_id': stay_id,
#                 'hour': hour,
#                 'heart_rate': np.random.normal(80, 15),
#                 'respiratory_rate': np.random.normal(20, 5),
#                 'systolic_bp': np.random.normal(120, 20),
#                 'diastolic_bp': np.random.normal(80, 10),
#                 'temperature': np.random.normal(37, 1),
#                 'spo2': np.random.normal(97, 2),
#                 'wbc': np.random.normal(8, 3),
#                 'lactate': np.random.normal(1.5, 0.8),
#                 'creatinine': np.random.normal(1.0, 0.3),
#                 'platelet': np.random.normal(250, 80),
#                 'bilirubin': np.random.normal(0.8, 0.4)
#             }
            
#             # 抗生素使用 (随机5%的时间点)
#             for j in range(1, 6):
#                 row[f'antibiotic_{j}'] = 1 if np.random.random() < 0.05 else 0
                
#             # 血管活性药物使用 (随机3%的时间点)
#             for j in range(1, 5):
#                 row[f'vasopressor_{j}'] = 1 if np.random.random() < 0.03 else 0
                
#             # 诊断ID (随机ICD代码)
#             icd_codes = [f"D{np.random.randint(1, 100)}" for _ in range(np.random.randint(1, 5))]
#             row['diagnosis_ids'] = ','.join(icd_codes)
            
#             # 文本嵌入 (减少嵌入维度，避免生成过大的随机数据)
#             embed_dim = min(embedding_dim, 32)  # 限制最大维度
            
#             # 预分配一个固定大小的嵌入向量
#             text_embed = np.random.normal(0, 0.1, size=embed_dim)
#             for i, val in enumerate(text_embed):
#                 row[f'text_embed_{i}'] = val
                
#             data.append(row)
    
#     # 创建特征DataFrame
#     print("创建患者特征DataFrame...")
#     patient_features = pd.DataFrame(data)
    
#     # 生成脓毒症标签 (约10%的患者会在某个时间点变为阳性)
#     print("生成脓毒症标签...")
#     sepsis_labels_data = []
#     for subject_id, hadm_id, stay_id in zip(subject_ids, hadm_ids, stay_ids):
#         has_sepsis = np.random.random() < 0.1
#         if has_sepsis:
#             # 随机选择发病时间
#             onset_hour = np.random.randint(12, n_hours)
            
#             # 生成SOFA评分轨迹（发病前逐渐上升，发病后保持高值）
#             for hour in range(n_hours):
#                 if hour < onset_hour - 12:
#                     sofa_score = np.random.randint(0, 2)
#                     sepsis3 = 0
#                 elif hour < onset_hour:
#                     sofa_score = np.random.randint(1, 6)
#                     sepsis3 = 0
#                 else:
#                     sofa_score = np.random.randint(4, 12)
#                     sepsis3 = 1
                
#                 row = {
#                     'subject_id': subject_id,
#                     'hadm_id': hadm_id,
#                     'stay_id': stay_id,
#                     'hour': hour,
#                     'sofa_score': sofa_score,
#                     'sepsis3': sepsis3,
#                     'sepsis_onset_time': onset_hour if hour == onset_hour else None
#                 }
#                 sepsis_labels_data.append(row)
#         else:
#             # 健康患者的SOFA评分较低
#             for hour in range(n_hours):
#                 row = {
#                     'subject_id': subject_id,
#                     'hadm_id': hadm_id,
#                     'stay_id': stay_id,
#                     'hour': hour,
#                     'sofa_score': np.random.randint(0, 3),
#                     'sepsis3': 0,
#                     'sepsis_onset_time': None
#                 }
#                 sepsis_labels_data.append(row)
    
#     # 创建DataFrame
#     sepsis_labels = pd.DataFrame(sepsis_labels_data)
    
#     # 确保列名一致性 ('hr' -> 'hour')
#     if 'hr' in sepsis_labels.columns and 'hour' not in sepsis_labels.columns:
#         sepsis_labels = sepsis_labels.rename(columns={'hr': 'hour'})
    
#     # 生成知识图谱嵌入
#     print("生成知识图谱嵌入...")
#     n_concepts = 1000
#     kg_embeddings = np.random.normal(0, 0.1, size=(n_concepts, embedding_dim))
    
#     # 生成时间轴信息
#     time_axis = {
#         "resolution": "1H",
#         "min_time": reference_time.strftime('%Y-%m-%d %H:%M:%S'),
#         "max_time": (reference_time + timedelta(hours=n_hours-1)).strftime('%Y-%m-%d %H:%M:%S')
#     }
    
#     return patient_features, sepsis_labels, kg_embeddings, time_axis

# 特征提取与变换
def preprocess_features(patient_data, feature_config):
    """
    预处理单个患者数据用于预测
    
    参数:
        patient_data: 患者数据DataFrame
        feature_config: 特征配置字典
        
    返回:
        vitals: 生命体征特征
        labs: 实验室检查特征
        drugs: 药物使用特征
        text_embed: 文本嵌入特征
    """
    print("预处理特征数据...")
    
    # 提取特征列
    vitals_cols = feature_config['vitals_columns']
    labs_cols = feature_config['labs_columns']
    drugs_cols = feature_config['drugs_columns']
    text_cols = feature_config['text_embed_columns']
    
    # 检查并过滤不存在的列
    available_vitals_cols = [col for col in vitals_cols if col in patient_data.columns]
    available_labs_cols = [col for col in labs_cols if col in patient_data.columns]
    available_drugs_cols = [col for col in drugs_cols if col in patient_data.columns]
    
    print(f"可用生命体征列: {available_vitals_cols} (缺失: {set(vitals_cols) - set(available_vitals_cols)})")
    print(f"可用实验室检查列: {available_labs_cols} (缺失: {set(labs_cols) - set(available_labs_cols)})")
    print(f"可用药物列: {available_drugs_cols} (缺失: {set(drugs_cols) - set(available_drugs_cols)})")
    
    # 创建特征矩阵
    n_samples = len(patient_data)
    
    # 处理生命体征特征 - 使用均值填充
    if available_vitals_cols:
        # 计算每个特征的均值，忽略NaN
        vitals_mean = patient_data[available_vitals_cols].mean(skipna=True)
        # 使用均值填充
        vitals = patient_data[available_vitals_cols].fillna(vitals_mean).values
        # 标准化
        vitals_scaler = StandardScaler()
        vitals = vitals_scaler.fit_transform(vitals)
        
        # 如果有缺失的列，添加零列
        if len(available_vitals_cols) < len(vitals_cols):
            missing_cols = np.zeros((n_samples, len(vitals_cols) - len(available_vitals_cols)))
            vitals = np.hstack([vitals, missing_cols])
    else:
        vitals = np.zeros((n_samples, len(vitals_cols)))
    
    # 处理实验室检查特征 - 使用均值填充
    if available_labs_cols:
        # 计算每个特征的均值，忽略NaN
        labs_mean = patient_data[available_labs_cols].mean(skipna=True)
        # 使用均值填充
        labs = patient_data[available_labs_cols].fillna(labs_mean).values
        # 标准化
        labs_scaler = StandardScaler()
        labs = labs_scaler.fit_transform(labs)
        
        # 如果有缺失的列，添加零列
        if len(available_labs_cols) < len(labs_cols):
            missing_cols = np.zeros((n_samples, len(labs_cols) - len(available_labs_cols)))
            labs = np.hstack([labs, missing_cols])
    else:
        labs = np.zeros((n_samples, len(labs_cols)))
    
    # 处理药物特征 - 二元特征，使用0填充
    if available_drugs_cols:
        drugs = patient_data[available_drugs_cols].fillna(0).values
        # 如果有缺失的列，添加零列
        if len(available_drugs_cols) < len(drugs_cols):
            missing_cols = np.zeros((n_samples, len(drugs_cols) - len(available_drugs_cols)))
            drugs = np.hstack([drugs, missing_cols])
    else:
        drugs = np.zeros((n_samples, len(drugs_cols)))
    
    # 处理文本嵌入特征
    available_text_cols = [col for col in text_cols if col in patient_data.columns]
    if available_text_cols:
        text_embeds = patient_data[available_text_cols].fillna(0).values
    else:
        print("警告：找不到文本嵌入列，使用全零向量")
        # 使用最小维度作为替代
        text_dim = feature_config.get('text_dim', 32)
        text_embeds = np.zeros((n_samples, text_dim))
    
    return vitals, labs, drugs, text_embeds