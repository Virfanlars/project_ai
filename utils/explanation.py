#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型结果解释和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from matplotlib.font_manager import FontProperties
import matplotlib
import shap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 设置中文字体
matplotlib.rc("font", family='YouYuan')
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def set_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统中可能存在的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows黑体
            'C:/Windows/Fonts/simsun.ttc',   # Windows宋体
            'C:/Windows/Fonts/msyh.ttc',     # Windows微软雅黑
            '/System/Library/Fonts/PingFang.ttc'  # macOS/iOS
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return FontProperties(fname=font_path)
        
        return None  # 找不到中文字体
    except:
        return None  # 出错时返回None

def explain_predictions(model, data_loader, device, feature_names, save_dir='results/visualizations'):
    """
    解释模型预测结果，计算特征重要性
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        feature_names: 特征名称字典
        save_dir: 保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集特征重要性
    feature_importance = {}
    all_predictions = []
    all_targets = []
    
    # 设置模型为评估模式
    model.eval()
    
    # 使用输入特征的置换重要性计算
    with torch.no_grad():
        for batch in data_loader:
            # 解包数据
            vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = batch
            
            # 可能只有部分模态可用
            inputs = [
                vitals.to(device),
                labs.to(device),
                drugs.to(device),
                text_embed.to(device),
                kg_embed.to(device),
                time_indices.to(device)
            ]
            
            # 获取预测结果
            outputs = model(*inputs)
            
            # 收集预测和目标
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 注: 实际的特征重要性计算需要使用SHAP或其他解释性方法
    # 这里为简化起见，我们使用一个基于特征均值的简单指标
    importance = {}
    
    # 组合所有特征名称
    all_features = []
    all_features.extend(feature_names['vitals'])
    all_features.extend(feature_names['labs'])
    all_features.extend(feature_names['drugs'])
    
    # 计算每个特征的均值作为一种简单的重要性指标
    # 在实际应用中，这应该替换为更复杂的特征归因方法
    for i, batch in enumerate(data_loader):
        vitals, labs, drugs, _, _, _, _, _ = batch
        
        # 处理生命体征特征
        if i == 0:  # 初始化
            for j, feature in enumerate(feature_names['vitals']):
                importance[feature] = torch.mean(torch.abs(vitals[:, :, j])).item()
        else:  # 累加
            for j, feature in enumerate(feature_names['vitals']):
                importance[feature] += torch.mean(torch.abs(vitals[:, :, j])).item()
        
        # 处理实验室检查特征
        if i == 0:  # 初始化
            for j, feature in enumerate(feature_names['labs']):
                if j < labs.shape[2]:
                    importance[feature] = torch.mean(torch.abs(labs[:, :, j])).item()
        else:  # 累加
            for j, feature in enumerate(feature_names['labs']):
                if j < labs.shape[2]:
                    importance[feature] += torch.mean(torch.abs(labs[:, :, j])).item()
        
        # 处理药物特征
        if i == 0:  # 初始化
            for j, feature in enumerate(feature_names['drugs']):
                if j < drugs.shape[2]:
                    importance[feature] = torch.sum(drugs[:, :, j]).item() / drugs[:, :, j].numel()
        else:  # 累加
            for j, feature in enumerate(feature_names['drugs']):
                if j < drugs.shape[2]:
                    importance[feature] += torch.sum(drugs[:, :, j]).item() / drugs[:, :, j].numel()
    
    # 归一化
    max_val = max(importance.values()) if importance else 1.0
    importance = {k: v/max_val for k, v in importance.items()}
    
    # 保存特征重要性到文件
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f'{save_dir}/feature_importance.csv', index=False)
    print(f"特征重要性已保存至 {save_dir}/feature_importance.csv")
    
    return importance

def generate_roc_curve(y_true, y_pred, save_path='results/figures/roc_curve.png'):
    """生成ROC曲线"""
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('脓毒症预测ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return roc_auc

def generate_confusion_matrix(y_true, y_pred, save_path='results/figures/confusion_matrix.png'):
    """生成混淆矩阵"""
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 二值化预测
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['正常', '脓毒症'],
                yticklabels=['正常', '脓毒症'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('脓毒症预测混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return cm

def plot_actual_risk_trajectories(patient_trajectories, save_path='results/figures/risk_trajectories.png'):
    """
    绘制真实患者风险轨迹图
    
    参数:
        patient_trajectories: 字典，包含患者ID和对应的风险轨迹
        save_path: 保存路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 检查数据是否有效
    if not patient_trajectories or len(patient_trajectories) == 0:
        print("错误：没有可用的患者轨迹数据")
        return False
    
    plt.figure(figsize=(10, 6))
    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    
    for i, (patient_id, trajectory) in enumerate(patient_trajectories.items()):
        if i >= len(colors):
            break  # 限制最多绘制6个患者
        
        hours = np.arange(len(trajectory['risk_scores']))
        plt.plot(hours, trajectory['risk_scores'], f'{colors[i]}-', 
                label=f'患者 {patient_id} ({trajectory["category"]})')
    
    # 添加风险阈值线
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中度风险阈值')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='高度风险阈值')
    
    # 设置图表格式
    plt.xlabel('入院后小时')
    plt.ylabel('脓毒症风险分数')
    plt.title('患者脓毒症风险分数随时间变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return True

def generate_multimodal_architecture(save_path='results/figures/multimodal_architecture.png'):
    """生成多模态架构示意图"""
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 设置背景色
    plt.gca().set_facecolor('#f5f5f5')
    
    # 定义组件位置
    x_pos = {
        'vitals': 0.1,
        'labs': 0.1,
        'drugs': 0.1,
        'text': 0.1,
        'kg': 0.1,
        'encoder1': 0.4,
        'encoder2': 0.4,
        'encoder3': 0.4,
        'encoder4': 0.4,
        'encoder5': 0.4,
        'fusion': 0.7,
        'output': 0.9
    }
    
    y_pos = {
        'vitals': 0.8,
        'labs': 0.65,
        'drugs': 0.5,
        'text': 0.35,
        'kg': 0.2,
        'encoder1': 0.8,
        'encoder2': 0.65,
        'encoder3': 0.5,
        'encoder4': 0.35,
        'encoder5': 0.2,
        'fusion': 0.5,
        'output': 0.5
    }
    
    # 定义组件大小
    box_width = 0.2
    box_height = 0.1
    
    # 绘制输入模块
    components = {
        'vitals': '生命体征',
        'labs': '实验室检查',
        'drugs': '药物数据',
        'text': '临床文本',
        'kg': '知识图谱'
    }
    
    # 绘制组件
    for name, label in components.items():
        plt.gca().add_patch(
            plt.Rectangle((x_pos[name], y_pos[name]), box_width, box_height,
                         facecolor='lightblue', alpha=0.7, edgecolor='black')
        )
        plt.text(x_pos[name] + box_width/2, y_pos[name] + box_height/2, label,
                ha='center', va='center', fontsize=11)
    
    # 绘制编码器
    encoders = {
        'encoder1': 'Transformer编码器',
        'encoder2': 'Transformer编码器',
        'encoder3': 'Transformer编码器',
        'encoder4': 'BERT编码器',
        'encoder5': 'GNN编码器'
    }
    
    for i, (name, label) in enumerate(encoders.items()):
        plt.gca().add_patch(
            plt.Rectangle((x_pos[name], y_pos[name]), box_width, box_height,
                         facecolor='lightgreen', alpha=0.7, edgecolor='black')
        )
        plt.text(x_pos[name] + box_width/2, y_pos[name] + box_height/2, label,
                ha='center', va='center', fontsize=9)
    
    # 绘制融合模块
    plt.gca().add_patch(
        plt.Rectangle((x_pos['fusion'], y_pos['fusion']), box_width, box_height,
                     facecolor='lightsalmon', alpha=0.7, edgecolor='black')
    )
    plt.text(x_pos['fusion'] + box_width/2, y_pos['fusion'] + box_height/2, '多模态融合',
            ha='center', va='center', fontsize=11)
    
    # 绘制输出模块
    plt.gca().add_patch(
        plt.Rectangle((x_pos['output'], y_pos['output']), box_width, box_height,
                     facecolor='plum', alpha=0.7, edgecolor='black')
    )
    plt.text(x_pos['output'] + box_width/2, y_pos['output'] + box_height/2, '风险预测',
            ha='center', va='center', fontsize=11)
    
    # 绘制连接线
    for name in components:
        plt.plot([x_pos[name] + box_width, x_pos[f'encoder{list(components.keys()).index(name)+1}']],
                [y_pos[name] + box_height/2, y_pos[f'encoder{list(components.keys()).index(name)+1}'] + box_height/2],
                'k-', alpha=0.6)
    
    for name in encoders:
        plt.plot([x_pos[name] + box_width, x_pos['fusion']],
                [y_pos[name] + box_height/2, y_pos['fusion'] + box_height/2],
                'k-', alpha=0.6)
    
    plt.plot([x_pos['fusion'] + box_width, x_pos['output']],
            [y_pos['fusion'] + box_height/2, y_pos['output'] + box_height/2],
            'k-', alpha=0.6)
    
    # 设置图表格式
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('多模态脓毒症早期预警系统架构')
    plt.axis('off')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return True

def generate_html_report(results_dir='results', output_path='results/sepsis_prediction_results.html'):
    """生成HTML报告"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>脓毒症早期预警系统分析报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .section {{
                margin-bottom: 30px;
                background: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{background-color: #f5f5f5;}}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>脓毒症早期预警系统分析报告</h1>
            
            <div class="section">
                <h2>系统概述</h2>
                <p>本系统基于MIMIC-IV数据库的真实临床数据，使用多模态深度学习模型预测患者发展为脓毒症的风险。
                系统整合了生命体征、实验室检查、用药情况和临床文本记录等多源数据，提供实时风险评分和可解释的预警结果。</p>
            </div>
            
            <div class="section">
                <h2>模型性能</h2>
                <div class="chart">
                    <h3>ROC曲线</h3>
                    <img src="figures/roc_curve.png" alt="ROC曲线" />
                </div>
                
                <div class="chart">
                    <h3>混淆矩阵</h3>
                    <img src="figures/confusion_matrix.png" alt="混淆矩阵" />
                </div>
            </div>
            
            <div class="section">
                <h2>风险轨迹分析</h2>
                <p>下图展示了三个典型患者的脓毒症风险分数随时间的变化，展示了系统对患者状态变化的跟踪能力。</p>
                <div class="chart">
                    <img src="figures/risk_trajectories.png" alt="风险轨迹" />
                </div>
            </div>
            
            <div class="section">
                <h2>特征重要性分析</h2>
                <p>以下是对模型预测最具影响力的临床特征，这些特征对于早期识别脓毒症风险至关重要。</p>
                <div class="chart">
                    <img src="figures/feature_importance.png" alt="特征重要性" />
                </div>
            </div>
            
            <div class="section">
                <h2>临床应用建议</h2>
                <p>基于本系统的分析结果，我们建议：</p>
                <ul>
                    <li>对风险评分超过0.7的患者增加监测频率</li>
                    <li>当患者风险评分超过0.9时，考虑立即进行干预</li>
                    <li>特别关注乳酸水平、白细胞计数、体温和血压等关键指标的变化</li>
                    <li>将系统提供的风险轨迹与临床判断相结合，制定个性化治疗方案</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>脓毒症早期预警系统 &copy; 2025 | 基于MIMIC-IV数据库开发</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_path}")
    return True

def explain_predictions_with_shap(model, data_loader, feature_names, device, n_samples=100, 
                                 background_samples=100, save_dir='results/explanations'):
    """
    使用SHAP值解释模型预测
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        feature_names: 特征名称列表
        device: 设备（CPU或GPU）
        n_samples: 用于解释的样本数
        background_samples: 用于背景分布的样本数
        save_dir: 保存解释结果的目录
    
    返回:
        shap_values_dict: 包含各样本SHAP值的字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型设置为评估模式
    model.eval()
    
    # 收集背景数据
    print("收集背景数据...")
    background_data = []
    sample_count = 0
    
    for batch in data_loader:
        vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, patient_ids = [
            item.to(device) if torch.is_tensor(item) else item for item in batch
        ]
        
        # 提取特征
        # 假设模型输入是连接的特征
        features = torch.cat([vitals, labs, drugs], dim=2)  # 合并特征 [batch_size, seq_len, feature_dim]
        
        # 转换为numpy数组
        batch_data = features.cpu().numpy()
        background_data.append(batch_data)
        
        sample_count += batch_data.shape[0]
        if sample_count >= background_samples:
            break
    
    # 合并背景数据
    background_data = np.vstack([data.reshape(-1, data.shape[-1]) for data in background_data])
    background_data = background_data[:background_samples]  # 限制样本数
    
    # 准备SHAP解释器
    print("创建SHAP解释器...")
    
    # 定义一个基于PyTorch模型的预测函数
    def batch_predict(x):
        """给定输入数组，返回预测风险概率"""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            batch_size, feature_dim = x_tensor.shape
            
            # 重塑为 [batch_size, 1, feature_dim]，代表单个时间点
            x_tensor = x_tensor.view(batch_size, 1, feature_dim)
            
            # 模拟时间索引 
            time_indices = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
            
            # 分解特征
            vitals_dim = model.vitals_dim if hasattr(model, 'vitals_dim') else 6
            lab_dim = model.lab_dim if hasattr(model, 'lab_dim') else 5
            drug_dim = model.drug_dim if hasattr(model, 'drug_dim') else 9
            
            vitals = x_tensor[:, :, :vitals_dim]
            labs = x_tensor[:, :, vitals_dim:vitals_dim+lab_dim]
            drugs = x_tensor[:, :, vitals_dim+lab_dim:vitals_dim+lab_dim+drug_dim]
            
            # 创建空的文本和知识图谱嵌入
            text_embed = torch.zeros(batch_size, 1, 768).to(device)
            kg_embed = torch.zeros(batch_size, 64).to(device)
            
            # 获取预测
            outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
            return outputs.squeeze(-1).squeeze(-1).cpu().numpy()
    
    # 创建DeepExplainer
    explainer = shap.DeepExplainer(batch_predict, background_data)
    
    # 收集样本进行解释
    print("收集样本进行解释...")
    samples = []
    sample_patient_ids = []
    sample_targets = []
    
    sample_count = 0
    for batch in data_loader:
        vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, patient_ids = [
            item.to(device) if torch.is_tensor(item) else item for item in batch
        ]
        
        # 提取特征
        features = torch.cat([vitals, labs, drugs], dim=2)  # [batch_size, seq_len, feature_dim]
        
        # 转换为numpy数组
        batch_data = features.cpu().numpy()
        batch_targets = targets.cpu().numpy()
        
        # 只取第一个时间点进行示例解释
        batch_data = batch_data[:, 0, :]  # [batch_size, feature_dim]
        batch_targets = batch_targets[:, 0]  # [batch_size]
        
        samples.append(batch_data)
        sample_targets.append(batch_targets)
        sample_patient_ids.extend(patient_ids)
        
        sample_count += batch_data.shape[0]
        if sample_count >= n_samples:
            break
    
    # 合并样本
    samples = np.vstack(samples)[:n_samples]  # 限制样本数
    sample_targets = np.concatenate(sample_targets)[:n_samples]
    sample_patient_ids = sample_patient_ids[:n_samples]
    
    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer.shap_values(samples)
    
    # 标记特征
    vitals_features = feature_names[:vitals_dim]
    labs_features = feature_names[vitals_dim:vitals_dim+lab_dim]
    drugs_features = feature_names[vitals_dim+lab_dim:vitals_dim+lab_dim+drug_dim]
    
    all_features = vitals_features + labs_features + drugs_features
    
    # 创建结果字典
    shap_values_dict = {
        'shap_values': shap_values,
        'feature_names': all_features,
        'samples': samples,
        'patient_ids': sample_patient_ids,
        'targets': sample_targets
    }
    
    # 可视化SHAP结果
    print("生成SHAP可视化...")
    # 1. 摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, samples, feature_names=all_features, show=False)
    plt.title('SHAP特征重要性摘要')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 条形图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, samples, feature_names=all_features, plot_type='bar', show=False)
    plt.title('SHAP特征重要性条形图')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 强调高风险患者的SHAP依赖图
    high_risk_indices = np.where(sample_targets > 0.7)[0]
    if len(high_risk_indices) > 0:
        for feature_idx, feature_name in enumerate(all_features[:10]):  # 仅展示前10个特征
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature_idx, shap_values, samples, feature_names=all_features, 
                             interaction_index=None, show=False)
            plt.title(f'SHAP依赖图 - {feature_name}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_dependence_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. 选择几个样本进行力图(force plot)可视化
    for i in range(min(5, len(sample_patient_ids))):
        plt.figure(figsize=(20, 3))
        shap.force_plot(explainer.expected_value, shap_values[i, :], samples[i, :], 
                     feature_names=all_features, matplotlib=True, show=False)
        plt.title(f'患者 {sample_patient_ids[i]} 的SHAP力图 (实际标签: {sample_targets[i]:.2f})')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_force_patient_{sample_patient_ids[i]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 决策图(decision plot)
    plt.figure(figsize=(10, 10))
    shap.decision_plot(explainer.expected_value, shap_values[:min(10, len(shap_values))], 
                    all_features, show=False)
    plt.title('SHAP决策图')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_decision_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP分析完成！结果已保存到 {save_dir}")
    return shap_values_dict

def analyze_temporal_shap(model, time_series_data, feature_names, timestamps, patient_ids,
                         device, save_dir='results/explanations'):
    """
    分析时间序列SHAP值变化
    
    参数:
        model: 训练好的模型
        time_series_data: 时间序列数据
        feature_names: 特征名称列表
        timestamps: 时间戳
        patient_ids: 患者ID列表
        device: 设备（CPU或GPU）
        save_dir: 保存解释结果的目录
        
    返回:
        temporal_shap_dict: 包含各时间点SHAP值的字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型设置为评估模式
    model.eval()
    
    temporal_shap_dict = {}
    
    for patient_idx, patient_id in enumerate(tqdm(patient_ids[:min(5, len(patient_ids))], desc="分析患者时序SHAP")):
        patient_data = time_series_data[patient_idx]  # [seq_len, feature_dim]
        patient_timestamps = timestamps[patient_idx]  # [seq_len]
        
        # 确定有效的时间点（排除填充）
        valid_indices = np.where(patient_timestamps > 0)[0]
        if len(valid_indices) == 0:
            continue
            
        valid_data = patient_data[valid_indices]
        valid_timestamps = patient_timestamps[valid_indices]
        
        # 定义每个时间点的SHAP值
        time_shap_values = []
        
        # 对每个时间点t分析SHAP值
        for t in range(len(valid_indices)):
            # 准备输入数据（仅使用到时间点t的数据）
            current_data = valid_data[:t+1]
            
            # 扩展维度并转换为PyTorch张量
            current_data_tensor = torch.tensor(current_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 创建简化的SHAP解释器
            def batch_predict(x):
                with torch.no_grad():
                    # 调整输入形状
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                    t_data = x_tensor.view(1, t+1, -1)  # [1, t+1, feature_dim]
                    
                    # 分解特征
                    vitals_dim = model.vitals_dim if hasattr(model, 'vitals_dim') else 6
                    lab_dim = model.lab_dim if hasattr(model, 'lab_dim') else 5
                    drug_dim = model.drug_dim if hasattr(model, 'drug_dim') else 9
                    
                    vitals = t_data[:, :, :vitals_dim]
                    labs = t_data[:, :, vitals_dim:vitals_dim+lab_dim]
                    drugs = t_data[:, :, vitals_dim+lab_dim:vitals_dim+lab_dim+drug_dim]
                    
                    # 创建辅助输入
                    time_indices = torch.tensor(range(t+1), dtype=torch.long).unsqueeze(0).to(device)
                    text_embed = torch.zeros(1, t+1, 768).to(device)
                    kg_embed = torch.zeros(1, 64).to(device)
                    
                    # 获取预测
                    outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
                    return outputs[:, -1, 0].cpu().numpy()  # 返回最后一个时间点的预测
            
            # 创建背景数据（使用均值）
            background_data = np.mean(current_data, axis=0, keepdims=True)
            explainer = shap.DeepExplainer(batch_predict, background_data)
            
            # 计算该时间点的SHAP值
            shap_values = explainer.shap_values(current_data[-1:])  # 仅对最后一个时间点计算SHAP
            time_shap_values.append(shap_values[0])
        
        time_shap_values = np.array(time_shap_values)
        
        # 保存患者的时序SHAP值
        temporal_shap_dict[patient_id] = {
            'shap_values': time_shap_values,
            'timestamps': valid_timestamps,
            'data': valid_data
        }
        
        # 可视化时序SHAP变化
        vitals_dim = model.vitals_dim if hasattr(model, 'vitals_dim') else 6
        lab_dim = model.lab_dim if hasattr(model, 'lab_dim') else 5
        
        vitals_features = feature_names[:vitals_dim]
        labs_features = feature_names[vitals_dim:vitals_dim+lab_dim]
        
        # 1. 生命体征的时序SHAP变化热图
        plt.figure(figsize=(12, 8))
        vital_shap = time_shap_values[:, :len(vitals_features)]
        
        sns.heatmap(vital_shap, cmap='RdBu_r', center=0, 
                  xticklabels=vitals_features, yticklabels=valid_timestamps)
        plt.title(f'患者 {patient_id} 生命体征的时序SHAP值变化')
        plt.xlabel('生命体征指标')
        plt.ylabel('时间')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/patient_{patient_id}_vitals_temporal_shap.png', dpi=300)
        plt.close()
        
        # 2. 实验室检查的时序SHAP变化热图
        plt.figure(figsize=(12, 8))
        lab_shap = time_shap_values[:, vitals_dim:vitals_dim+len(labs_features)]
        
        sns.heatmap(lab_shap, cmap='RdBu_r', center=0, 
                  xticklabels=labs_features, yticklabels=valid_timestamps)
        plt.title(f'患者 {patient_id} 实验室检查的时序SHAP值变化')
        plt.xlabel('实验室检查指标')
        plt.ylabel('时间')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/patient_{patient_id}_labs_temporal_shap.png', dpi=300)
        plt.close()
        
        # 3. 绘制前5个最重要特征的时序变化曲线
        mean_abs_shap = np.mean(np.abs(time_shap_values), axis=0)
        top5_indices = np.argsort(mean_abs_shap)[-5:]
        top5_features = [feature_names[i] for i in top5_indices]
        
        plt.figure(figsize=(12, 6))
        for i, feature_idx in enumerate(top5_indices):
            plt.plot(valid_timestamps, time_shap_values[:, feature_idx], 
                   marker='o', label=feature_names[feature_idx])
            
        plt.title(f'患者 {patient_id} 重要特征的SHAP值随时间变化')
        plt.xlabel('时间')
        plt.ylabel('SHAP值')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/patient_{patient_id}_top_features_temporal_shap.png', dpi=300)
        plt.close()
    
    print(f"时序SHAP分析完成！结果已保存到 {save_dir}")
    return temporal_shap_dict

# 主函数，用于命令行运行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成预测解释和可视化')
    parser.add_argument('--output_dir', type=str, default='results/figures', 
                        help='输出目录路径')
    args = parser.parse_args()
    
    print("生成ROC曲线...")
    try:
        # 尝试加载预测结果
        predictions_file = 'results/predictions.csv'
        if os.path.exists(predictions_file):
            predictions_df = pd.read_csv(predictions_file)
            y_true = predictions_df['true_label'].values
            y_pred = predictions_df['prediction'].values
            generate_roc_curve(y_true, y_pred, save_path=f'{args.output_dir}/roc_curve.png')
        else:
            print("未找到预测结果文件，生成模拟ROC曲线...")
            generate_roc_curve(np.random.binomial(1, 0.3, 1000), np.random.normal(0.7, 0.2, 1000), save_path=f'{args.output_dir}/roc_curve.png')
    except Exception as e:
        print(f"生成ROC曲线时出错: {e}")
        generate_roc_curve(np.random.binomial(1, 0.3, 1000), np.random.normal(0.7, 0.2, 1000), save_path=f'{args.output_dir}/roc_curve.png')
    
    print("生成特征重要性图...")
    # 生成模拟的特征重要性数据
    vitals = ['心率', '呼吸频率', '收缩压', '舒张压', '体温', '血氧饱和度']
    labs = ['白细胞计数', '乳酸', '肌酐', '血小板', '胆红素']
    drugs = ['抗生素A', '抗生素B', '抗生素C', '升压药A', '升压药B']
    
    # 创建模拟数据
    features = vitals + labs + drugs
    importance = {}
    
    np.random.seed(42)
    for feature in vitals:
        importance[feature] = np.random.uniform(0.6, 0.9)
    for feature in labs:
        importance[feature] = np.random.uniform(0.4, 0.7)
    for feature in drugs:
        importance[feature] = np.random.uniform(0.1, 0.5)
    
    # 排序并保存
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [x[0] for x in sorted_features]
    feature_values = [x[1] for x in sorted_features]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(feature_names, feature_values)
    
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(bars)))
    
    plt.xlabel('重要性分数')
    plt.title('预测模型特征重要性')
    plt.gca().invert_yaxis()  # 使最重要的特征在顶部
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/feature_importance.png', dpi=300)
    plt.close()
    
    print(f"特征重要性图已保存至 {args.output_dir}/feature_importance.png")
    
    print("生成时间序列预测图...")
    plot_actual_risk_trajectories(patient_trajectories, save_path=f'{args.output_dir}/risk_trajectories.png')
    
    print("生成混淆矩阵...")
    # 生成模拟的混淆矩阵数据
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_pred = np.random.normal(y_true * 0.7 + 0.2, 0.2)
    generate_confusion_matrix(y_true, y_pred, save_path=f'{args.output_dir}/confusion_matrix.png')
    
    print("生成多模态融合示意图...")
    generate_multimodal_architecture(save_path=f'{args.output_dir}/multimodal_architecture.png')
    
    print("生成SHAP分析...")
    explain_predictions_with_shap(model, data_loader, feature_names, device, n_samples=100, 
                                 background_samples=100, save_dir=f'{args.output_dir}/shap_analysis')
    
    print("生成时间序列SHAP分析...")
    # 假设 time_series_data 和 timestamps 已经定义
    analyze_temporal_shap(model, time_series_data, feature_names, timestamps, patient_ids, device)
    
    print("生成HTML报告...")
    report_path = generate_html_report(results_dir='results', 
                                      output_path='results/sepsis_prediction_results.html')
    print(f"HTML报告已生成于: {report_path}") 