import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

from utils.data_loading import load_structured_data
from utils.dataset import SepsisDataset
from models.fusion.multimodal_model import SepsisTransformerModel
from utils.explanation import explain_patient_prediction
from utils.visualization import visualize_risk_timeline
from config import MODEL_CONFIG, FEATURE_CONFIG

def analyze_patient(patient_id, model_path='best_model.pt'):
    """分析单个患者的风险情况和解释"""
    # 1. 加载数据
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 2. 筛选目标患者数据
    patient_data = patient_features[patient_features['subject_id'] == patient_id]
    patient_labels = sepsis_labels[sepsis_labels['subject_id'] == patient_id]
    
    if len(patient_data) == 0:
        print(f"未找到患者 {patient_id} 的数据")
        return
    
    # 3. 创建模型
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. 创建特征
    dataset = SepsisDataset(
        patient_features=patient_features,
        sepsis_labels=sepsis_labels,
        kg_embeddings=kg_embeddings,
        time_axis=time_axis,
        feature_config=FEATURE_CONFIG,
        max_seq_len=len(patient_data)  # 不需要填充或截断
    )
    
    # 找到患者在数据集中的索引
    patient_idx = list(dataset.patient_ids).index(patient_id)
    vitals, labs, drugs, text_embed, kg_embed, time_indices, labels, onset_time = dataset[patient_idx]
    
    # 5. 预测风险
    with torch.no_grad():
        vitals = vitals.unsqueeze(0).to(device)
        labs = labs.unsqueeze(0).to(device)
        drugs = drugs.unsqueeze(0).to(device)
        text_embed = text_embed.unsqueeze(0).to(device)
        kg_embed = kg_embed.unsqueeze(0).to(device)
        time_indices = time_indices.unsqueeze(0).to(device)
        
        risk_scores = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
        risk_scores = risk_scores.squeeze().cpu().numpy()
    
    # 6. 计算每个时间点的特征重要性
    feature_importance_over_time = {}
    feature_names = (
        FEATURE_CONFIG['vitals_columns'] + 
        FEATURE_CONFIG['labs_columns'] + 
        FEATURE_CONFIG['drugs_columns']
    )
    
    import shap
    explainer = shap.DeepExplainer(model, [
        vitals[:1], labs[:1], drugs[:1], text_embed[:1], kg_embed[:1], time_indices[:1]
    ])
    
    # 使用最小批次大小的数据计算SHAP值
    shap_values = explainer.shap_values([
        vitals, labs, drugs, text_embed, kg_embed, time_indices
    ])
    
    # 合并所有特征的SHAP值
    hours = time_indices.squeeze().cpu().numpy()
    for t_idx, hour in enumerate(hours):
        if hour == 0 and t_idx > 0:  # 跳过填充的时间点
            continue
            
        # 提取该时间点所有特征的重要性
        t_importances = []
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx < len(FEATURE_CONFIG['vitals_columns']):
                # 生命体征特征
                importance = shap_values[0][0, t_idx, feat_idx]
            elif feat_idx < len(FEATURE_CONFIG['vitals_columns']) + len(FEATURE_CONFIG['labs_columns']):
                # 实验室检查特征
                lab_idx = feat_idx - len(FEATURE_CONFIG['vitals_columns'])
                importance = shap_values[1][0, t_idx, lab_idx]
            else:
                # 药物特征
                drug_idx = feat_idx - len(FEATURE_CONFIG['vitals_columns']) - len(FEATURE_CONFIG['labs_columns'])
                importance = shap_values[2][0, t_idx, drug_idx]
            
            t_importances.append((feat_name, importance))
        
        # 按重要性绝对值排序
        t_importances.sort(key=lambda x: abs(x[1]), reverse=True)
        feature_importance_over_time[hour] = t_importances
    
    # 7. 可视化风险随时间变化
    # 获取实际时间戳
    timestamps = patient_data['hour'].values
    
    # 获取临床事件
    clinical_events = []
    # 例如: 抗生素使用、机械通气等事件
    for hour in patient_data['hour']:
        row = patient_data[patient_data['hour'] == hour]
        
        # 检查是否存在重要临床事件
        if 'antibiotic_1' in row.columns and row['antibiotic_1'].values[0] > 0:
            clinical_events.append((hour, "开始抗生素治疗"))
        
        # 其他临床事件...
    
    # 生成可视化
    visualize_risk_timeline(
        patient_id=patient_id,
        timestamps=timestamps,
        risk_scores=risk_scores,
        feature_importance_over_time=feature_importance_over_time,
        clinical_events=clinical_events
    )
    
    # 8. 生成患者报告
    onset_str = f"实际脓毒症发作时间: {onset_time}小时" if onset_time else "无脓毒症发作记录"
    
    # 获取最高风险时间点
    max_risk_idx = np.argmax(risk_scores)
    max_risk_time = timestamps[max_risk_idx]
    max_risk_score = risk_scores[max_risk_idx]
    
    # 获取该时间点的重要特征
    if max_risk_time in feature_importance_over_time:
        top_features = feature_importance_over_time[max_risk_time][:5]
        feature_str = "\n".join([f"  - {name}: {score:.4f}" for name, score in top_features])
    else:
        feature_str = "无法获取特征重要性"
    
    # 打印报告
    print(f"\n患者 {patient_id} 风险分析报告:")
    print("-" * 50)
    print(f"最高风险评分: {max_risk_score:.4f} (时间点: {max_risk_time}小时)")
    print(f"最重要的风险因素:")
    print(feature_str)
    print(f"\n{onset_str}")
    print("-" * 50)
    print(f"风险时间线和特征重要性可视化已保存在 'patient_{patient_id}_timeline.png'")
    
    # 返回结果
    return {
        'patient_id': patient_id,
        'risk_scores': risk_scores.tolist(),
        'timestamps': timestamps.tolist(),
        'max_risk_time': int(max_risk_time),
        'max_risk_score': float(max_risk_score),
        'feature_importance': {str(k): v for k, v in feature_importance_over_time.items()}
    }

if __name__ == "__main__":
    # 示例: 分析患者ID为12345的风险情况
    analyze_patient(12345) 