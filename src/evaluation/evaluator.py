#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估模块
用于评估模型性能和生成评估指标
"""

import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate_model(model, model_path, data_loader, kg_embeddings, device, output_dir='./output'):
    """
    评估模型性能
    
    Args:
        model: 模型实例
        model_path: 模型权重文件路径
        data_loader: 测试数据加载器
        kg_embeddings: 知识图谱嵌入字典
        device: 评估设备
        output_dir: 输出目录
        
    Returns:
        dict: 评估指标
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 获取实体嵌入
    entity_embeddings = torch.tensor(kg_embeddings['entity_embeddings'], dtype=torch.float32).to(device)
    
    # 用于存储预测和真实标签
    all_preds = []
    all_labels = []
    all_patient_ids = []
    all_times = []
    
    # 评估模型
    logger.info("开始评估模型...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # 获取批次数据
            vitals = batch['vitals'].to(device)
            labs = batch['labs'].to(device)
            drugs = batch['drugs'].to(device)
            labels = batch['labels'].to(device)
            time_indices = batch['time_indices'].to(device)
            
            # 创建注意力掩码 - 使用更宽松的掩码条件
            vitals_valid = vitals.sum(dim=2) != 0  # 至少有一个生命体征特征不为零
            labs_valid = labs.sum(dim=2) != 0      # 至少有一个实验室值特征不为零
            drugs_valid = drugs.sum(dim=2) != 0    # 至少有一个药物特征不为零
            
            # 至少一组特征有效，该位置就视为有效
            valid_positions = vitals_valid | labs_valid | drugs_valid
            
            # 反转为掩码（True表示填充位置）
            attention_mask = ~valid_positions
            
            # 确保至少有一个非掩码位置
            if attention_mask.all(dim=1).any():
                # 找出所有位置都被掩码的样本
                fully_masked = attention_mask.all(dim=1)
                # 对这些样本，将第一个位置设为非掩码
                attention_mask[fully_masked, 0] = False
            
            # 为每个样本随机选择一个知识图谱嵌入
            batch_size = vitals.size(0)
            kg_indices = torch.randint(0, len(entity_embeddings), (batch_size,)).to(device)
            kg_embeds = entity_embeddings[kg_indices]
            
            # 简化处理：使用零向量作为文本嵌入
            text_embeds = torch.zeros(batch_size, 768).to(device)
            
            # 前向传播
            outputs = model(vitals, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
            
            # 只考虑非填充位置
            mask = ~attention_mask
            masked_outputs = outputs * mask
            masked_labels = labels * mask
            
            # 收集预测和标签
            all_preds.append(masked_outputs.cpu().numpy())
            all_labels.append(masked_labels.cpu().numpy())
            
            # 记录患者ID（这里使用批次索引和样本索引作为ID）
            for i in range(batch_size):
                all_patient_ids.extend([f"{batch_idx}_{i}"] * vitals.size(1))
                all_times.extend(time_indices[i].cpu().numpy())
    
    # 将所有预测和标签合并
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    
    # 过滤掉填充位置（NaN值）
    valid_indices = ~np.isnan(all_preds) & ~np.isnan(all_labels)
    all_preds = all_preds[valid_indices]
    all_labels = all_labels[valid_indices]
    
    # 计算评估指标
    # 1. ROC曲线下面积
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        logger.warning("计算AUROC时出错，可能是标签中只有一个类别")
        auroc = 0.5  # 使用随机分类器的性能作为默认值
    
    # 2. PR曲线下面积
    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        auprc = auc(recall, precision)
    except:
        logger.warning("计算AUPRC时出错")
        auprc = 0.5
    
    # 3. 使用0.5作为阈值的指标
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    
    # 处理可能的零除错误
    try:
        precision_val = precision_score(all_labels, binary_preds)
    except:
        precision_val = 0.0
        
    try:
        recall_val = recall_score(all_labels, binary_preds)
    except:
        recall_val = 0.0
        
    try:
        f1 = f1_score(all_labels, binary_preds)
    except:
        f1 = 0.0
    
    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 5. 提前预警时间（小时）- 使用更合理的计算方法
    # 创建DataFrame来分析提前预警时间
    results_df = pd.DataFrame({
        'patient_id': np.array(all_patient_ids)[valid_indices],
        'time': np.array(all_times)[valid_indices],
        'prediction': all_preds,
        'label': all_labels
    })
    
    # 计算每个患者的提前预警时间
    early_warning_hours = []
    false_alarm_rates = []  # 记录假警报率
    
    # 对结果按患者ID和时间排序
    results_df = results_df.sort_values(['patient_id', 'time'])
    
    # 使用更保守的预测阈值，减少假阳性
    prediction_threshold = 0.6
    
    for patient_id in results_df['patient_id'].unique():
        patient_data = results_df[results_df['patient_id'] == patient_id]
        
        # 确保时间正确排序
        patient_data = patient_data.sort_values('time')
        
        # 如果患者确实有脓毒症
        if patient_data['label'].max() > 0:
            # 找到第一个脓毒症标签为1的时间
            sepsis_onset_rows = patient_data[patient_data['label'] > 0]
            if not sepsis_onset_rows.empty:
                sepsis_time = sepsis_onset_rows['time'].min()
                
                # 找到第一个预测值超过阈值的时间
                # 为防止短暂的假警报，要求连续两个时间点预测值都超过阈值
                # 先找到预测值超过阈值的所有行
                alarm_rows = patient_data[patient_data['prediction'] > prediction_threshold]
                
                if not alarm_rows.empty:
                    # 转换为NumPy数组以便更好地处理连续性检查
                    alarm_times = alarm_rows['time'].values
                    all_times = patient_data['time'].values
                    
                    # 找到第一个有效警报（连续两个时间点都超过阈值）
                    first_valid_alarm_time = None
                    for i in range(len(alarm_times)):
                        current_time = alarm_times[i]
                        # 找到当前时间点在所有时间点中的索引
                        time_idx = np.where(all_times == current_time)[0][0]
                        
                        # 如果不是最后一个时间点，检查下一个时间点是否也是警报
                        if time_idx < len(all_times) - 1:
                            next_time = all_times[time_idx + 1]
                            if next_time in alarm_times:
                                first_valid_alarm_time = current_time
                                break
                    
                    # 如果找到了有效警报，并且发生在脓毒症之前
                    if first_valid_alarm_time is not None and first_valid_alarm_time < sepsis_time:
                        warning_time = sepsis_time - first_valid_alarm_time
                        # 过滤掉不合理的预警时间（超过24小时可能是假警报）
                        if 0 < warning_time <= 24:
                            early_warning_hours.append(warning_time)
        else:
            # 对于没有脓毒症的患者，检查是否有假警报
            false_alarms = patient_data[patient_data['prediction'] > prediction_threshold]
            if not false_alarms.empty:
                false_alarm_rates.append(1)  # 记录有假警报
            else:
                false_alarm_rates.append(0)  # 记录无假警报
    
    # 计算平均提前预警时间和假警报率
    mean_early_warning = np.mean(early_warning_hours) if early_warning_hours else 0
    median_early_warning = np.median(early_warning_hours) if early_warning_hours else 0
    false_alarm_rate = np.mean(false_alarm_rates) if false_alarm_rates else 0
    
    # 收集所有指标
    metrics = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'mean_early_warning_hours': float(mean_early_warning),
        'median_early_warning_hours': float(median_early_warning),
        'false_alarm_rate': float(false_alarm_rate),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    # 保存详细结果，用于可视化
    detailed_results = results_df.to_dict(orient='records')
    detailed_results_path = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # 保存患者轨迹数据，用于可视化
    results_df.to_csv(os.path.join(output_dir, 'patient_trajectories.csv'), index=False)
    
    # 保存评估指标
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 打印评估结果
    logger.info("评估完成，结果如下：")
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"准确率: {accuracy:.4f}")
    logger.info(f"精确率: {precision_val:.4f}")
    logger.info(f"召回率: {recall_val:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    logger.info(f"特异性: {specificity:.4f}")
    logger.info(f"假警报率: {false_alarm_rate:.4f}")
    logger.info(f"平均提前预警时间: {mean_early_warning:.2f}小时")
    logger.info(f"中位提前预警时间: {median_early_warning:.2f}小时")
    logger.info(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return metrics


def evaluate_feature_importance(model, data_loader, kg_embeddings, device, output_dir='./output'):
    """
    评估特征重要性
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        kg_embeddings: 知识图谱嵌入
        device: 评估设备
        output_dir: 输出目录
        
    Returns:
        dict: 特征重要性分数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 获取实体嵌入
    entity_embeddings = torch.tensor(kg_embeddings['entity_embeddings'], dtype=torch.float32).to(device)
    
    # 获取特征名称
    vitals_features = data_loader.dataset.vitals_features
    lab_features = data_loader.dataset.lab_features
    drug_features = data_loader.dataset.drug_features
    
    # 初始化特征重要性分数
    feature_importance = {
        'vitals': {feature: 0.0 for feature in vitals_features},
        'labs': {feature: 0.0 for feature in lab_features},
        'drugs': {feature: 0.0 for feature in drug_features},
        'kg': 0.0
    }
    
    # 计数器
    n_samples = 0
    
    # 使用排列重要性方法评估特征重要性
    logger.info("计算特征重要性...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Feature Importance"):
            # 获取批次数据
            vitals = batch['vitals'].to(device)
            labs = batch['labs'].to(device)
            drugs = batch['drugs'].to(device)
            labels = batch['labels'].to(device)
            time_indices = batch['time_indices'].to(device)
            
            # 创建注意力掩码 - 使用更宽松的掩码条件
            vitals_valid = vitals.sum(dim=2) != 0  # 至少有一个生命体征特征不为零
            labs_valid = labs.sum(dim=2) != 0      # 至少有一个实验室值特征不为零
            drugs_valid = drugs.sum(dim=2) != 0    # 至少有一个药物特征不为零
            
            # 至少一组特征有效，该位置就视为有效
            valid_positions = vitals_valid | labs_valid | drugs_valid
            
            # 反转为掩码（True表示填充位置）
            attention_mask = ~valid_positions
            
            # 确保至少有一个非掩码位置
            if attention_mask.all(dim=1).any():
                # 找出所有位置都被掩码的样本
                fully_masked = attention_mask.all(dim=1)
                # 对这些样本，将第一个位置设为非掩码
                attention_mask[fully_masked, 0] = False
            
            # 为每个样本随机选择一个知识图谱嵌入
            batch_size = vitals.size(0)
            kg_indices = torch.randint(0, len(entity_embeddings), (batch_size,)).to(device)
            kg_embeds = entity_embeddings[kg_indices]
            
            # 简化处理：使用零向量作为文本嵌入
            text_embeds = torch.zeros(batch_size, 768).to(device)
            
            # 基准预测
            base_outputs = model(vitals, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
            
            # 只考虑非填充位置
            mask = ~attention_mask
            base_outputs = base_outputs * mask
            
            # 对每个特征进行排列并计算重要性
            # 1. 生命体征特征
            for i, feature in enumerate(vitals_features):
                # 复制数据
                vitals_permuted = vitals.clone()
                
                # 排列特征（在批次内随机打乱）
                for b in range(batch_size):
                    perm_idx = torch.randperm(vitals.size(1))
                    vitals_permuted[b, :, i] = vitals[b, perm_idx, i]
                
                # 使用排列后的特征进行预测
                perm_outputs = model(vitals_permuted, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
                perm_outputs = perm_outputs * mask
                
                # 计算性能下降
                importance = torch.abs(base_outputs - perm_outputs).mean().item()
                feature_importance['vitals'][feature] += importance
            
            # 2. 实验室值特征
            for i, feature in enumerate(lab_features):
                # 复制数据
                labs_permuted = labs.clone()
                
                # 排列特征
                for b in range(batch_size):
                    perm_idx = torch.randperm(labs.size(1))
                    labs_permuted[b, :, i] = labs[b, perm_idx, i]
                
                # 使用排列后的特征进行预测
                perm_outputs = model(vitals, labs_permuted, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
                perm_outputs = perm_outputs * mask
                
                # 计算性能下降
                importance = torch.abs(base_outputs - perm_outputs).mean().item()
                feature_importance['labs'][feature] += importance
            
            # 3. 药物使用特征
            for i, feature in enumerate(drug_features):
                # 复制数据
                drugs_permuted = drugs.clone()
                
                # 排列特征
                for b in range(batch_size):
                    perm_idx = torch.randperm(drugs.size(1))
                    drugs_permuted[b, :, i] = drugs[b, perm_idx, i]
                
                # 使用排列后的特征进行预测
                perm_outputs = model(vitals, labs, drugs_permuted, text_embeds, kg_embeds, time_indices, attention_mask)
                perm_outputs = perm_outputs * mask
                
                # 计算性能下降
                importance = torch.abs(base_outputs - perm_outputs).mean().item()
                feature_importance['drugs'][feature] += importance
            
            # 4. 知识图谱嵌入
            # 随机打乱知识图谱嵌入
            perm_kg_indices = torch.randperm(len(entity_embeddings))[:batch_size].to(device)
            perm_kg_embeds = entity_embeddings[perm_kg_indices]
            
            # 使用排列后的特征进行预测
            perm_outputs = model(vitals, labs, drugs, text_embeds, perm_kg_embeds, time_indices, attention_mask)
            perm_outputs = perm_outputs * mask
            
            # 计算性能下降
            importance = torch.abs(base_outputs - perm_outputs).mean().item()
            feature_importance['kg'] += importance
            
            # 更新计数器
            n_samples += 1
    
    # 计算平均重要性
    for feature in feature_importance['vitals']:
        feature_importance['vitals'][feature] /= n_samples
    
    for feature in feature_importance['labs']:
        feature_importance['labs'][feature] /= n_samples
    
    for feature in feature_importance['drugs']:
        feature_importance['drugs'][feature] /= n_samples
    
    feature_importance['kg'] /= n_samples
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    logger.info("特征重要性计算完成")
    
    return feature_importance 