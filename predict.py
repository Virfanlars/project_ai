# 实时预测脚本
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os

from models.fusion.multimodal_model import SepsisTransformerModel
from utils.data_loading import preprocess_features
from utils.explanation import explain_predictions
from utils.visualization import visualize_risk_timeline, generate_patient_report
from config import MODEL_CONFIG, FEATURE_CONFIG

def predict_patient_risk(patient_data, kg_embeddings=None, model_path='models/best_model.pt'):
    """
    预测单个患者的脓毒症风险
    
    参数:
        patient_data: 患者时序数据DataFrame
        kg_embeddings: 知识图谱嵌入
        model_path: 模型权重路径
    
    返回:
        risk_scores: 风险评分
        explanations: 解释
    """
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. 预处理特征
    vitals, labs, drugs, text_embed = preprocess_features(patient_data, FEATURE_CONFIG)
    
    # 3. 提取诊断ID (用于知识图谱嵌入)
    diagnosis_ids = []
    if 'diagnosis_ids' in patient_data.columns:
        diag_values = patient_data['diagnosis_ids'].dropna()
        if not diag_values.empty:
            diag_str = diag_values.iloc[0]
            if isinstance(diag_str, str):
                diagnosis_ids = diag_str.split(',')
    
    # 4. 获取知识图谱嵌入
    if kg_embeddings is not None:
        kg_embed = get_kg_embedding(diagnosis_ids, kg_embeddings)
    else:
        # 使用零向量
        kg_embed = np.zeros(MODEL_CONFIG['kg_dim'])
    
    # 5. 构建时间索引
    time_indices = torch.tensor(patient_data['hour'].map(
        lambda x: pd.to_datetime(x).hour if isinstance(x, str) else 0
    ).values, dtype=torch.long)
    
    # 6. 转换为张量并添加批次维度
    vitals_tensor = torch.FloatTensor(vitals).unsqueeze(0).to(device)
    labs_tensor = torch.FloatTensor(labs).unsqueeze(0).to(device)
    drugs_tensor = torch.FloatTensor(drugs).unsqueeze(0).to(device)
    text_embed_tensor = torch.FloatTensor(text_embed).unsqueeze(0).to(device)
    kg_embed_tensor = torch.FloatTensor(kg_embed).repeat(len(vitals), 1).unsqueeze(0).to(device)
    time_indices_tensor = time_indices.unsqueeze(0).to(device)
    
    # 7. 预测风险
    with torch.no_grad():
        outputs = model(
            vitals_tensor, 
            labs_tensor, 
            drugs_tensor, 
            text_embed_tensor, 
            kg_embed_tensor, 
            time_indices_tensor
        )
        risk_scores = outputs.squeeze().cpu().numpy()
    
    # 8. 生成解释
    batch_data = [
        vitals_tensor, labs_tensor, drugs_tensor, 
        text_embed_tensor, kg_embed_tensor, time_indices_tensor
    ]
    
    feature_names = (
        FEATURE_CONFIG['vitals_columns'] + 
        FEATURE_CONFIG['labs_columns'] + 
        FEATURE_CONFIG['drugs_columns']
    )
    
    explanations = explain_predictions(model, batch_data, feature_names)
    
    return risk_scores, explanations

def get_kg_embedding(diagnosis_ids, kg_embeddings):
    """获取知识图谱嵌入"""
    # 提取诊断的嵌入
    diagnosis_embeddings = []
    for d in diagnosis_ids:
        if d.startswith('D'):
            try:
                d_idx = int(d.lstrip('D'))
                if d_idx < len(kg_embeddings):
                    diagnosis_embeddings.append(kg_embeddings[d_idx])
            except (ValueError, IndexError):
                continue
    
    # 计算平均嵌入
    if diagnosis_embeddings:
        return np.mean(diagnosis_embeddings, axis=0)
    else:
        # 返回零向量
        return np.zeros(kg_embeddings.shape[1])

def main():
    parser = argparse.ArgumentParser(description='预测患者脓毒症风险')
    parser.add_argument('--patient_id', type=str, required=True, help='患者ID')
    parser.add_argument('--data_file', type=str, default='data/processed/patient_features.csv', help='患者特征文件')
    parser.add_argument('--kg_file', type=str, default='data/processed/kg_embeddings.npy', help='知识图谱嵌入文件')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', help='模型权重文件')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    patient_features = pd.read_csv(args.data_file)
    kg_embeddings = np.load(args.kg_file)
    
    # 提取目标患者数据
    patient_data = patient_features[patient_features['subject_id'] == int(args.patient_id)]
    
    if len(patient_data) == 0:
        print(f"错误：找不到患者 {args.patient_id} 的数据")
        return
    
    # 按时间排序
    patient_data = patient_data.sort_values('hour')
    
    # 预测风险
    risk_scores, explanations = predict_patient_risk(
        patient_data, 
        kg_embeddings, 
        args.model_path
    )
    
    # 可视化风险曲线
    timestamps = patient_data['hour'].values
    
    visualize_risk_timeline(
        patient_id=args.patient_id,
        timestamps=timestamps,
        risk_scores=risk_scores,
        feature_importance_over_time=explanations,
        clinical_events=None  # 可以添加实际的临床事件
    )
    
    # 生成风险报告
    sepsis_label = None  # 获取实际标签（如果有）
    sepsis_onset_time = None
    
    # 尝试加载标签数据
    try:
        labels_file = 'data/processed/sepsis_labels.csv'
        if os.path.exists(labels_file):
            sepsis_labels = pd.read_csv(labels_file)
            patient_labels = sepsis_labels[sepsis_labels['subject_id'] == int(args.patient_id)]
            
            if not patient_labels.empty and 1 in patient_labels['sepsis_label'].values:
                sepsis_label = True
                onset_idx = patient_labels['sepsis_label'].values.argmax()
                sepsis_onset_time = patient_labels['hour'].iloc[onset_idx]
            else:
                sepsis_label = False
    except:
        pass
    
    report = generate_patient_report(
        patient_id=args.patient_id,
        risk_scores=risk_scores,
        timestamps=timestamps,
        feature_importance=explanations,
        sepsis_label=sepsis_label,
        sepsis_onset_time=sepsis_onset_time
    )
    
    # 保存报告
    report_path = os.path.join(args.output_dir, f'patient_{args.patient_id}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到 {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    main() 