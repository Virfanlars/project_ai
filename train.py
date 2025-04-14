import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import time
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
matplotlib.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc

from utils.data_loading import load_structured_data
from utils.dataset import SepsisDataset
from models.fusion.multimodal_model import SepsisTransformerModel
from config import FEATURE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

class EarlyStopping:
    """
    早停类，用于防止过拟合
    """
    def __init__(self, patience=5, verbose=False, delta=0, path='models/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """保存模型"""
        if self.verbose:
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})，保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_sepsis_model(model, train_loader, val_loader, num_epochs=100, 
                      learning_rate=0.001, weight_decay=1e-5, device='cuda',
                      gradient_clip=1.0, early_stopping_patience=10, 
                      lr_scheduler=True):
    """
    训练脓毒症预警模型，包含更多高级训练技术
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 初始学习率
        weight_decay: 权重衰减系数
        device: 训练设备
        gradient_clip: 梯度裁剪阈值
        early_stopping_patience: 早停耐心值
        lr_scheduler: 是否使用学习率调度器
    """
    print(f"开始训练，轮数: {num_epochs}, 学习率: {learning_rate}, 设备: {device}")
    start_time = time.time()
    
    # 移动模型到设备
    model = model.to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # 初始化学习率调度器
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, verbose=True, path='models/best_model.pt'
    )
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'auroc': [],
        'auprc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for i, batch in enumerate(train_loader):
            # 解包批次数据
            vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                item.to(device) if torch.is_tensor(item) else item for item in batch
            ]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
            loss = criterion(outputs.squeeze(), targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # 每50个批次打印一次进度
            if (i + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        avg_train_loss = train_loss / batch_count
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                    item.to(device) if torch.is_tensor(item) else item for item in batch
                ]
                
                outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                batch_count += 1
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / batch_count
        history['val_loss'].append(avg_val_loss)
        
        # 计算性能指标
        auroc = roc_auc_score(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
        history['auroc'].append(auroc)
        history['auprc'].append(auprc)
        
        # 打印训练和验证结果
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, LR: {current_lr}')
        
        # 使用学习率调度器
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"早停在轮次 {epoch+1}")
            break
    
    # 训练完成，记录总时间
    training_time = time.time() - start_time
    print(f"训练完成，总耗时: {training_time/60:.2f} 分钟")
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 保存训练历史
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f)
    
    return model

def plot_training_history(history):
    """绘制训练历史图表"""
    os.makedirs('results/figures', exist_ok=True)
    
    # 训练和验证损失
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # AUROC和AUPRC
    plt.subplot(1, 2, 2)
    plt.plot(history['auroc'], label='AUROC')
    plt.plot(history['auprc'], label='AUPRC')
    plt.title('验证集性能指标')
    plt.xlabel('轮次')
    plt.ylabel('分数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/figures/training_history.png', dpi=300)
    plt.close()
    
    # 学习率
    plt.figure(figsize=(6, 4))
    plt.plot(history['lr'])
    plt.title('学习率变化')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('results/figures/learning_rate.png', dpi=300)
    plt.close()

def create_weighted_sampler(dataset):
    """创建加权采样器来处理不平衡数据"""
    # 获取所有标签
    targets = []
    for idx in range(len(dataset)):
        _, _, _, _, _, _, target, _ = dataset[idx]
        targets.append(target.mean().item())  # 使用平均值，因为标签可能是时间序列
    
    # 计算类别权重
    positive_samples = sum(targets)
    negative_samples = len(targets) - positive_samples
    
    if negative_samples == 0 or positive_samples == 0:
        return None  # 如果只有一个类别，不使用加权采样
    
    # 少数类别得到更高的权重
    weights = torch.FloatTensor([
        1.0 / negative_samples if t < 0.5 else 1.0 / positive_samples for t in targets
    ])
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练脓毒症预警模型')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['weight_decay'],
                        help='权重衰减系数')
    parser.add_argument('--early_stopping', type=int, default=TRAIN_CONFIG['early_stopping_patience'],
                        help='早停耐心值')
    parser.add_argument('--gradient_clip', type=float, default=TRAIN_CONFIG.get('gradient_clip', 1.0),
                        help='梯度裁剪阈值')
    parser.add_argument('--use_lr_scheduler', action='store_true', 
                        help='使用学习率调度器')
    parser.add_argument('--use_sampler', action='store_true',
                        help='使用加权采样处理不平衡数据')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # 加载数据
    print("加载处理好的数据...")
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 创建数据集
    print("创建数据集...")
    dataset = SepsisDataset(
        patient_features=patient_features,
        sepsis_labels=sepsis_labels,
        kg_embeddings=kg_embeddings,
        time_axis=time_axis,
        feature_config=FEATURE_CONFIG
    )
    
    # 数据集划分
    print("划分训练、验证和测试集...")
    # 使用固定的随机种子以确保可重复性
    generator = torch.Generator().manual_seed(42)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"训练集: {train_size} 样本, 验证集: {val_size} 样本, 测试集: {test_size} 样本")
    
    # 创建采样器（如果需要）
    sampler = None
    if args.use_sampler:
        print("创建加权采样器处理不平衡数据...")
        sampler = create_weighted_sampler(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None),  # 如果使用采样器，不需要shuffle
        sampler=sampler,
        num_workers=4,  # 使用多进程加载数据
        pin_memory=True  # 加速数据传输到GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    print(f"初始化模型，隐藏维度: {MODEL_CONFIG['hidden_dim']}, 注意力头数: {MODEL_CONFIG['num_heads']}, 层数: {MODEL_CONFIG['num_layers']}")
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    print(f"开始训练模型，总轮数: {args.epochs}")
    model = train_sepsis_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping,
        lr_scheduler=args.use_lr_scheduler
    )
    
    # 保存数据集划分信息，用于后续分析
    torch.save({
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
    }, 'models/dataset_splits.pt')
    
    # 评估最佳模型
    print("加载最佳模型进行评估...")
    model.load_state_dict(torch.load('models/best_model.pt'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                item.to(device) if torch.is_tensor(item) else item for item in batch
            ]
            
            outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算性能指标
    auroc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    
    print("\n最终测试集性能:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('测试集ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig('results/figures/test_roc_curve.png', dpi=300)
    
    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {auprc:.3f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('测试集PR曲线')
    plt.legend(loc="upper right")
    plt.savefig('results/figures/test_pr_curve.png', dpi=300)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_label': all_targets,
        'prediction': all_preds
    })
    predictions_df.to_csv('results/predictions.csv', index=False)
    
    print(f"评估完成，结果已保存至 results 目录")
    print("模型训练与评估完成!")

if __name__ == "__main__":
    main()