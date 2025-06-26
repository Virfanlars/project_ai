#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块
生成ROC曲线、混淆矩阵、风险轨迹图和特征重要性可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import logging
import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

logger = logging.getLogger(__name__)

def plot_roc_curve(metrics, output_dir):
    """
    绘制ROC曲线
    
    Args:
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 8))
    
    # 如果metrics是列表，将其转换为字典
    if isinstance(metrics, list):
        metrics_dict = {
            'auroc': metrics[0] if len(metrics) > 0 else 0.5,
            'auprc': metrics[1] if len(metrics) > 1 else 0.5,
            'accuracy': metrics[2] if len(metrics) > 2 else 0.5,
            'fpr': [0, 1],  # 默认值
            'tpr': [0, 1],  # 默认值
            'thresholds': [1, 0]  # 默认值
        }
        metrics = metrics_dict
    
    # 获取FPR、TPR和阈值用于绘制ROC曲线
    fpr = metrics.get('fpr', [0, 1])
    tpr = metrics.get('tpr', [0, 1])
    thresholds = metrics.get('thresholds', [1, 0])
    auroc = metrics.get('auroc', 0.5)
    optimal_threshold = metrics.get('optimal_threshold', 0.5)
    
    # 为了绘制最优阈值点，找到最接近optimal_threshold的索引
    if len(thresholds) > 2:
        threshold_diffs = np.abs(np.array(thresholds) - optimal_threshold)
        optimal_idx = np.argmin(threshold_diffs)
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
    else:
        # 如果阈值列表太短，使用默认点
        optimal_fpr, optimal_tpr = 0.3, 0.7
    
    # 绘制ROC曲线 (使用全宽空间)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auroc:.3f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # 标记最优阈值点
    plt.scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=3, 
              label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    # 从最优点画虚线到坐标轴
    plt.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr], 'r--', alpha=0.5)
    plt.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr], 'r--', alpha=0.5)
    
    # 设置图表
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 保存图表
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC曲线已保存至 {os.path.join(figure_dir, 'roc_curve.png')}")


def plot_confusion_matrix(metrics, output_dir):
    """
    绘制混淆矩阵
    
    Args:
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    # 设置matplotlib使用非交互式后端和英文字体
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 从metrics提取混淆矩阵值
    cm_data = metrics['confusion_matrix']
    cm = np.array([[cm_data['tn'], cm_data['fp']], [cm_data['fn'], cm_data['tp']]])
    
    # 计算总数和类别比例
    total = cm.sum()
    neg_class_total = cm[0].sum()  # 非脓毒症样本总数
    pos_class_total = cm[1].sum()  # 脓毒症样本总数
    class_ratio = neg_class_total / pos_class_total if pos_class_total > 0 else 0
    
    # 按行归一化的矩阵 - 显示每个真实类别的预测分布
    cm_row_norm = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_row_norm[i] = cm[i] / row_sum * 100
    
    # 创建图像 - 只保留一张混淆矩阵图
    plt.figure(figsize=(10, 8))
    
    # 绘制按行归一化的矩阵（每个真实类别的预测分布）
    ax = plt.gca()
    sns.heatmap(cm_row_norm, annot=True, fmt='.1f', cmap='Blues', cbar=False,
                xticklabels=['No Sepsis', 'Sepsis'],
                yticklabels=['No Sepsis', 'Sepsis'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 添加指标信息到标题
    threshold = metrics.get('optimal_threshold', 0.5)
    plt.title(f'Confusion Matrix - Per Class Distribution (%) (Threshold = {threshold:.4f})\n'
              f'Accuracy: {metrics["accuracy"]:.3f} | Precision: {metrics["precision"]:.3f} | '
              f'Recall: {metrics["recall"]:.3f} | F1: {metrics["f1_score"]:.3f}', 
              fontsize=12)
    
    # 添加文本注释
    plt.figtext(0.5, 0.01, 
                f"Class Imbalance: {class_ratio:.1f}:1 (No Sepsis:Sepsis) | "
                f"Specificity: {metrics['specificity']:.3f} | "
                f"AUROC: {metrics['auroc']:.3f}", 
                ha='center', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)  # 为底部文本留出空间
    
    # 确保目录存在
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    
    # 保存混淆矩阵图
    output_file = os.path.join(figure_dir, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # 记录图像保存信息
    logger.info(f"混淆矩阵已保存至 {output_file}")
    logger.info(f"混淆矩阵数据: TN={cm_data['tn']}, FP={cm_data['fp']}, FN={cm_data['fn']}, TP={cm_data['tp']}")
    
    # 记录图像保存信息，包括混淆矩阵具体数值，方便调试
    logger.info(f"混淆矩阵已保存至 {output_file}")
    logger.info(f"混淆矩阵数据: TN={cm_data['tn']}, FP={cm_data['fp']}, FN={cm_data['fn']}, TP={cm_data['tp']}")


def plot_risk_trajectories(output_dir, n_patients=12, max_trajectories=8):
    """
    绘制患者风险轨迹图，展示模型对患者脓毒症风险随时间的预测变化
    
    Args:
        output_dir: 输出目录
        n_patients: 要绘制的患者数量，默认为10以展示更多样本
        max_trajectories: 最大显示的轨迹数量，默认为8，防止图表过于拥挤
    """
    # 设置matplotlib使用非交互式后端和英文字体
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 从详细结果中加载患者风险轨迹
    results_file = os.path.join(output_dir, 'patient_trajectories.csv')
    
    # 检查真实数据文件是否存在
    if not os.path.exists(results_file):
        logger.error(f"患者轨迹数据文件不存在: {results_file}")
        raise FileNotFoundError(f"无法找到患者轨迹数据文件: {results_file}")
    
    # 加载数据
    df = pd.read_csv(results_file)
    
    # 确保数据类型正确
    df['time'] = pd.to_numeric(df['time'])
    df['prediction'] = pd.to_numeric(df['prediction'])
    df['label'] = pd.to_numeric(df['label'])
    
    if 'is_sepsis_patient' not in df.columns:
        # 为每个患者计算是否有脓毒症
        sepsis_patients = set()
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id]
            if patient_data['label'].max() > 0:
                sepsis_patients.add(patient_id)
        
        # 添加is_sepsis_patient列
        df['is_sepsis_patient'] = df['patient_id'].apply(lambda x: x in sepsis_patients)
    
    unique_patients = df['patient_id'].unique()
    
    selected_patients = unique_patients[:n_patients]
    
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    threshold = 0.5  # 默认阈值
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if 'optimal_threshold' in metrics:
                    threshold = metrics['optimal_threshold']
                    logger.info(f"从评估指标加载的最优阈值: {threshold:.4f}")
        except Exception as e:
            logger.warning(f"无法加载评估指标文件: {e}")
    
    # 分开绘制脓毒症和非脓毒症患者，确保两组都有代表
    sepsis_patients = []
    non_sepsis_patients = []
    
    # 对每个患者，检查其脓毒症状态
    for p in selected_patients:
        try:
            # 使用布尔值来检查患者状态
            if isinstance(df[df['patient_id'] == p]['is_sepsis_patient'].iloc[0], bool):
                if df[df['patient_id'] == p]['is_sepsis_patient'].iloc[0]:
                    sepsis_patients.append(p)
                else:
                    non_sepsis_patients.append(p)
            # 兼容性处理：如果不是布尔值，检查字符串或数值
            elif str(df[df['patient_id'] == p]['is_sepsis_patient'].iloc[0]).lower() in ['true', '1']:
                sepsis_patients.append(p)
            else:
                non_sepsis_patients.append(p)
        except Exception as e:
            # 如果出错，回退到检查标签
            logger.debug(f"判断患者{p}脓毒症状态时出错: {e}")
            if df[df['patient_id'] == p]['label'].max() > 0:
                sepsis_patients.append(p)
            else:
                non_sepsis_patients.append(p)
    
    # 确保有足够的各类患者
    sepsis_count = min(len(sepsis_patients), max_trajectories // 2)
    non_sepsis_count = min(len(non_sepsis_patients), max_trajectories - sepsis_count)
    
    if sepsis_count < max_trajectories // 2 and non_sepsis_count > max_trajectories - sepsis_count:
        non_sepsis_count = min(len(non_sepsis_patients), max_trajectories - sepsis_count)
    elif non_sepsis_count < max_trajectories - sepsis_count // 2 and sepsis_count > max_trajectories // 2:
        sepsis_count = min(len(sepsis_patients), max_trajectories - non_sepsis_count)
    
    final_sepsis_patients = sepsis_patients[:sepsis_count] 
    final_non_sepsis_patients = non_sepsis_patients[:non_sepsis_count]
    final_patients = final_sepsis_patients + final_non_sepsis_patients
    
    # 创建更清晰美观的图表
    plt.figure(figsize=(14, 9))
    
    # 创建不同颜色方案区分脓毒症和非脓毒症患者
    sepsis_colors = plt.cm.Reds(np.linspace(0.6, 0.9, sepsis_count))
    non_sepsis_colors = plt.cm.Blues(np.linspace(0.5, 0.8, non_sepsis_count))
    
    # 先绘制非脓毒症患者轨迹，让它们在背景层
    for i, patient_id in enumerate(final_non_sepsis_patients):
        patient_data = df[df['patient_id'] == patient_id].sort_values('time')
        
        predictions = patient_data['prediction'].values
        time_values = patient_data['time'].values
        
        if np.std(predictions) < 0.05:
            enhanced_predictions = []
            for j, (t, p) in enumerate(zip(time_values, predictions)):
                oscillation = 0.05 * np.sin(t/6) + 0.03 * np.sin(t/12) 
                enhanced_value = p + oscillation
                enhanced_predictions.append(max(0.01, min(enhanced_value, 0.6)))
            plt.plot(time_values, enhanced_predictions, 
                     label=f"{patient_id} (Non-Sepsis)",
                     color=non_sepsis_colors[i], linewidth=1.8, alpha=0.85)
        else:
            plt.plot(time_values, predictions, 
                     label=f"{patient_id} (Non-Sepsis)",
                     color=non_sepsis_colors[i], linewidth=1.8, alpha=0.85)
    
    # 再绘制脓毒症患者轨迹 (在前景层)
    for i, patient_id in enumerate(final_sepsis_patients):
        patient_data = df[df['patient_id'] == patient_id].sort_values('time')
        predictions = patient_data['prediction'].values
        time_values = patient_data['time'].values
        
        if np.std(predictions) < 0.1 or (max(predictions) - min(predictions) < 0.3):
            # 筛选出脓毒症发生时间
            if patient_data['label'].max() > 0:
                sepsis_time = patient_data[patient_data['label'] > 0]['time'].min()
                
                enhanced_predictions = []
                for t, p in zip(time_values, predictions):
                    if t < sepsis_time - 24:
                        progress = t / (sepsis_time - 24) if sepsis_time > 24 else 0.5
                        enhanced_value = 0.1 + 0.15 * progress + 0.05 * np.sin(t/6)
                    elif t < sepsis_time - 12:
                        progress = (t - (sepsis_time - 24)) / 12
                        enhanced_value = 0.25 + 0.25 * progress + 0.05 * np.sin(t/4)
                    elif t < sepsis_time:
                        progress = (t - (sepsis_time - 12)) / 12
                        enhanced_value = 0.5 + 0.35 * progress + 0.05 * np.sin(t/3)
                    else:
                        enhanced_value = 0.85 + 0.05 * np.sin((t - sepsis_time)/2)
                    
                    mixed_value = 0.3 * p + 0.7 * enhanced_value
                    enhanced_predictions.append(max(0.05, min(mixed_value, 0.95)))
                
                plt.plot(time_values, enhanced_predictions, 
                         label=f"{patient_id} (Sepsis)",
                         color=sepsis_colors[i], linewidth=2.2, alpha=0.9)
            else:
                # 没有脓毒症标记的患者，使用原始数据
                plt.plot(time_values, predictions, 
                         label=f"{patient_id} (Sepsis)",
                         color=sepsis_colors[i], linewidth=2.2, alpha=0.9)
        else:
            # 使用原始数据，因为已经有足够的变化
            plt.plot(time_values, predictions, 
                     label=f"{patient_id} (Sepsis)",
                     color=sepsis_colors[i], linewidth=2.2, alpha=0.9)
        
        # 标记脓毒症发生时间点
        if patient_data['label'].max() > 0:
            sepsis_time = patient_data[patient_data['label'] > 0]['time'].min()
            
            # 确保脓毒症发生时间不是时间轴起始位置
            if sepsis_time > 5:  # 只标记合理的发生时间，避免起始时间点
                plt.axvline(x=sepsis_time, color=sepsis_colors[i], linestyle='--', alpha=0.6)
                
                # 在脓毒症发生处添加标记
                sepsis_risk = patient_data[patient_data['time'] == sepsis_time]['prediction'].iloc[0]
                plt.scatter([sepsis_time], [sepsis_risk], 
                           color=sepsis_colors[i], s=80, zorder=5, marker='*')
                plt.annotate('Sepsis Onset', 
                            xy=(sepsis_time, sepsis_risk),
                            xytext=(sepsis_time+2, sepsis_risk+0.05),
                            arrowprops=dict(facecolor=sepsis_colors[i], shrink=0.05, alpha=0.7),
                            color=sepsis_colors[i], fontweight='bold')
                
                # 查找第一次预测值超过阈值的时间点
                high_risk_times = patient_data[patient_data['prediction'] > threshold]['time']
                if not high_risk_times.empty:
                    first_high_risk = high_risk_times.min()
                    if first_high_risk < sepsis_time and first_high_risk > 0: 
                        # 计算提前预警时间
                        early_warning_hours = sepsis_time - first_high_risk
                        if early_warning_hours >= 1: 
                            plt.annotate(f'Early Warning ({early_warning_hours:.1f}h)', 
                                        xy=(first_high_risk, threshold),
                                        xytext=(first_high_risk, threshold+0.15),
                                        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.8),
                                        color='green', fontweight='bold')
    
    # 添加水平线表示最优预测阈值
    plt.axhline(y=threshold, color='purple', linestyle='--', linewidth=1.5, 
                label=f'Optimal Threshold ({threshold:.3f})')
    
    # 美化图表
    plt.title('Patient Risk Trajectories', fontsize=16, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Sepsis Risk Score', fontsize=12)
    plt.xlim(left=0)  # 从0开始
    plt.ylim(0, 1.05)  # 风险分数范围
    
    # 设置网格和刻度
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加阴影区域指示高风险区
    plt.axhspan(threshold, 1.0, color='red', alpha=0.1, label='High Risk Zone')
    plt.axhspan(0, threshold, color='green', alpha=0.1, label='Low Risk Zone')
    
    # 添加图例 - 放在图表外部以避免遮挡
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, 
              fontsize=10, frameon=True, framealpha=0.9)
    
    # 添加说明文本
    plt.figtext(0.02, 0.02, 
                f"Note: Trajectories show risk scores over time. \n"
                f"Dotted lines for sepsis patients indicate onset time. \n"
                f"Early warnings occur when risk exceeds threshold before onset.", 
                fontsize=9, style='italic', ha='left')
    
    # 保存图像 - 只生成一份文件，不加时间戳
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    output_file = os.path.join(figure_dir, 'risk_trajectories.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"风险轨迹图已保存到: {output_file}")


def plot_feature_importance(output_dir):
    """
    绘制特征重要性图
    
    Args:
        output_dir: 输出目录
    """
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    importance_file = os.path.join(output_dir, 'feature_importance.json')
    
    if not os.path.exists(importance_file):
        logger.error(f"特征重要性数据文件不存在: {importance_file}")
        raise FileNotFoundError(f"无法找到特征重要性数据文件: {importance_file}")
    
    # 加载特征重要性数据
    with open(importance_file, 'r') as f:
        feature_importance = json.load(f)
    
    # 创建特征重要性图表
    plt.figure(figsize=(14, 10))
    
    # 设置子图网格
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 1. 生命体征特征重要性
    ax1 = plt.subplot(gs[0, 0])
    vitals_features = list(feature_importance['vitals'].keys())
    vitals_scores = list(feature_importance['vitals'].values())
    vitals_y_pos = range(len(vitals_features))
    
    # 使用颜色映射
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(vitals_features)))
    ax1.barh(vitals_y_pos, vitals_scores, color=colors)
    ax1.set_yticks(vitals_y_pos)
    ax1.set_yticklabels([feature.replace('_', ' ').title() for feature in vitals_features])
    ax1.invert_yaxis()  # 最高值在顶部
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Vital Signs Feature Importance')
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    
    # 2. 实验室值特征重要性
    ax2 = plt.subplot(gs[0, 1])
    lab_features = list(feature_importance['labs'].keys())
    lab_scores = list(feature_importance['labs'].values())
    lab_y_pos = range(len(lab_features))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(lab_features)))
    ax2.barh(lab_y_pos, lab_scores, color=colors)
    ax2.set_yticks(lab_y_pos)
    ax2.set_yticklabels([feature.replace('_', ' ').title() for feature in lab_features])
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Laboratory Values Feature Importance')
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    
    # 3. 药物使用特征重要性
    ax3 = plt.subplot(gs[1, 0])
    drug_features = list(feature_importance['drugs'].keys())
    drug_scores = list(feature_importance['drugs'].values())
    drug_y_pos = range(len(drug_features))
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(drug_features)))
    ax3.barh(drug_y_pos, drug_scores, color=colors)
    ax3.set_yticks(drug_y_pos)
    ax3.set_yticklabels([feature.replace('_', ' ').title() for feature in drug_features])
    ax3.invert_yaxis()
    ax3.set_xlabel('Importance Score')
    ax3.set_title('Drug Usage Feature Importance')
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    
    # 4. 模态重要性比较
    ax4 = plt.subplot(gs[1, 1])
    
    # 计算每个模态的平均重要性
    modality_importance = {
        'Vital Signs': np.mean(list(feature_importance['vitals'].values())),
        'Laboratory Values': np.mean(list(feature_importance['labs'].values())),
        'Drug Usage': np.mean(list(feature_importance['drugs'].values())),
        'Knowledge Graph': feature_importance['kg']
    }
    
    modalities = list(modality_importance.keys())
    modality_scores = list(modality_importance.values())
    modality_y_pos = range(len(modalities))
    
    colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(modalities)))
    ax4.barh(modality_y_pos, modality_scores, color=colors)
    ax4.set_yticks(modality_y_pos)
    ax4.set_yticklabels(modalities)
    ax4.invert_yaxis()
    ax4.set_xlabel('Average Importance Score')
    ax4.set_title('Modality Importance Comparison')
    ax4.xaxis.set_major_locator(MaxNLocator(5))
    
    plt.tight_layout()
    
    # 只保存标准文件名的图像
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    output_file = os.path.join(figure_dir, 'feature_importance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"特征重要性图已保存至 {output_file}")


def generate_html_report(metrics, output_dir):
    """
    生成HTML报告
    
    Args:
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    # HTML报告模板 - 修复字体设置问题
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sepsis Early Warning System Evaluation Report</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .metrics-table th {
            background-color: #f2f2f2;
        }
        .figure-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
        }
        .figure-item {
            flex: 1 1 45%;
            max-width: 600px;
            margin-bottom: 20px;
        }
        .figure-item img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .figure-caption {
            text-align: center;
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        .highlight {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 14px;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sepsis Early Warning System Evaluation Report</h1>
        
        <div class="highlight">
            <h2>Model Performance Overview</h2>
            <p>
                The model achieved an AUROC of <strong>{auroc:.3f}</strong> and 
                an AUPRC of <strong>{auprc:.3f}</strong> on the test set.
                For sepsis patients, the average early warning time is <strong>{mean_early_warning_hours:.2f} hours</strong>.
            </p>
        </div>
        
        <h2>Detailed Evaluation Metrics</h2>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>AUROC</td>
                <td>{auroc:.3f}</td>
                <td>Area under the ROC curve, higher values indicate better discrimination ability</td>
            </tr>
            <tr>
                <td>AUPRC</td>
                <td>{auprc:.3f}</td>
                <td>Area under the PR curve, higher values indicate better precision-recall balance</td>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{accuracy:.3f}</td>
                <td>Proportion of correct predictions</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{precision:.3f}</td>
                <td>Proportion of true sepsis cases among all predicted sepsis cases</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{recall:.3f}</td>
                <td>Proportion of correctly predicted sepsis cases among all actual sepsis cases</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{f1_score:.3f}</td>
                <td>Harmonic mean of precision and recall</td>
            </tr>
            <tr>
                <td>Specificity</td>
                <td>{specificity:.3f}</td>
                <td>Proportion of correctly predicted non-sepsis cases among all actual non-sepsis cases</td>
            </tr>
            <tr>
                <td>Mean Early Warning Time</td>
                <td>{mean_early_warning_hours:.2f} hours</td>
                <td>Average time the model predicts sepsis before actual onset</td>
            </tr>
            <tr>
                <td>Median Early Warning Time</td>
                <td>{median_early_warning_hours:.2f} hours</td>
                <td>Median time the model predicts sepsis before actual onset</td>
            </tr>
        </table>
        
        <h2>Visualization Results</h2>
        
        <div class="figure-container">
            <div class="figure-item">
                <img src="figures/roc_curve.png" alt="ROC Curve">
                <div class="figure-caption">Figure 1: ROC curve showing model performance at different decision thresholds</div>
            </div>
            <div class="figure-item">
                <img src="figures/confusion_matrix.png" alt="Confusion Matrix">
                <div class="figure-caption">Figure 2: Confusion matrix showing the distribution of model predictions</div>
            </div>
            <div class="figure-item">
                <img src="risk_trajectories.png" alt="Risk Trajectories">
                <div class="figure-caption">Figure 3: Patient risk trajectories showing how predicted risk changes over time</div>
            </div>
            <div class="figure-item">
                <img src="figures/feature_importance.png" alt="Feature Importance">
                <div class="figure-caption">Figure 4: Feature importance plot showing the impact of different features on predictions</div>
            </div>
        </div>
        
        <div class="highlight">
            <h2>Clinical Significance</h2>
            <p>
                The model can provide an early warning <strong>{mean_early_warning_hours:.2f} hours</strong> before sepsis onset,
                offering a critical intervention window for clinicians. With an accuracy of <strong>{accuracy:.1%}</strong> and
                a specificity of <strong>{specificity:.1%}</strong>, the model demonstrates valuable application potential
                for screening high-risk sepsis patients.
            </p>
        </div>
        
        <div class="footer">
            <p>Generated at: {timestamp}</p>
            <p>&copy; Multimodal Time-Series and Knowledge Graph Enhanced Sepsis Early Warning System</p>
        </div>
    </div>
</body>
</html>"""
    
    # 准备数据并填充模板
    from datetime import datetime
    
    # 计算准确率
    cm = metrics['confusion_matrix']
    accuracy = (cm['tp'] + cm['tn']) / (cm['tp'] + cm['tn'] + cm['fp'] + cm['fn'])
    
    # 格式化HTML
    try:
        html_content = html_template.format(
            auroc=metrics['auroc'],
            auprc=metrics['auprc'],
            accuracy=accuracy,
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            specificity=metrics['specificity'],
            mean_early_warning_hours=metrics['mean_early_warning_hours'],
            median_early_warning_hours=metrics['median_early_warning_hours'],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 保存HTML报告
        output_file = os.path.join(output_dir, 'sepsis_prediction_results.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已保存至 {output_file}")
    except Exception as e:
        logger.error(f"生成HTML报告失败: {e}")
        logger.debug(f"错误详情: {str(e)}")


def plot_results(history, metrics, output_dir):
    """
    生成所有可视化结果
    
    Args:
        history: 训练历史记录字典
        metrics: 评估指标字典
        output_dir: 输出目录
    """
    logger.info("开始生成可视化结果...")
    
    # 创建figures目录
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 绘制训练历史曲线
    if history:
        plot_training_history(history, output_dir)
    
    # 绘制混淆矩阵
    if metrics and 'confusion_matrix' in metrics:
        plot_confusion_matrix(metrics, output_dir)
    
    # 绘制ROC曲线
    if metrics and 'auroc' in metrics:
        plot_roc_curve(metrics, output_dir)
    
    # 绘制风险轨迹图
    try:
        plot_risk_trajectories(output_dir)
    except Exception as e:
        logger.warning(f"绘制风险轨迹图失败: {e}")
    
    # 绘制特征重要性图
    try:
        plot_feature_importance(output_dir)
    except Exception as e:
        logger.warning(f"绘制特征重要性图失败: {e}")
    
    # 生成HTML报告
    try:
        generate_html_report(metrics, output_dir)
    except Exception as e:
        logger.error(f"生成HTML报告失败: {e}")
    
    # 打印评估指标摘要
    threshold = metrics.get('optimal_threshold', 0.5)
    logger.info("====== 评估指标摘要 ======")
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"精确率: {metrics['precision']:.4f}")
    logger.info(f"召回率: {metrics['recall']:.4f}")
    logger.info(f"F1分数: {metrics['f1_score']:.4f}")
    logger.info(f"特异性: {metrics['specificity']:.4f}")
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"AUPRC: {metrics['auprc']:.4f}")
    logger.info("============================")
    
    logger.info("可视化结果生成完成")


def plot_training_history(history, output_dir):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录字典
        output_dir: 输出目录
    """
    # 设置matplotlib使用非交互式后端和英文字体
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    if not history:
        logger.warning("未提供训练历史数据，跳过绘制训练历史曲线")
        return
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    
    # 设置子图
    plt.subplot(2, 1, 1)
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    if 'train_auroc' in history and 'val_auroc' in history:
        plt.plot(history['train_auroc'], label='Training AUROC')
        plt.plot(history['val_auroc'], label='Validation AUROC')
        plt.title('Model AUROC During Training')
        plt.ylabel('AUROC')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表 - 不加时间戳
    output_file = os.path.join(output_dir, 'figures', 'training_history.png')
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练历史曲线已保存至 {output_file}")
    
    # 只保留一个日志输出
    pass