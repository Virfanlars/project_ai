#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练模块
实现训练循环、早停和模型保存
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    早停类，用于在验证损失不再改善时停止训练
    """
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        """
        初始化早停
        
        Args:
            patience: 在多少个epoch内验证损失不改善就停止
            delta: 被认为是改善的最小差异
            path: 模型保存路径
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model):
        """
        在每个epoch结束时调用
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
            
        Returns:
            bool: 是否早停
        """
        # 如果更好，则保存模型
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model):
        """
        保存模型
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
        """
        logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, data_loaders, kg_embeddings, device, learning_rate=0.001, epochs=50, patience=10, output_dir='./output'):
    """
    训练脓毒症预测模型
    
    Args:
        model: 模型实例
        data_loaders: 包含训练集和验证集的DataLoader字典
        kg_embeddings: 知识图谱嵌入字典
        device: 训练设备
        learning_rate: 学习率
        epochs: 训练轮数
        patience: 早停耐心值
        output_dir: 输出目录
        
    Returns:
        dict: 训练历史记录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置损失函数、优化器和学习率调度器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 设置早停
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    # 准备训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auroc': [],
        'val_auroc': []
    }
    
    # 提取训练集和验证集
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # 获取实体嵌入
    entity_embeddings = torch.tensor(kg_embeddings['entity_embeddings'], dtype=torch.float32).to(device)
    
    # 记录开始时间
    start_time_total = time.time()
    
    # 训练循环
    logger.info(f"开始训练，共{epochs}个epoch")
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            try:
                # 获取批次数据
                vitals = batch['vitals'].to(device)
                labs = batch['labs'].to(device)
                drugs = batch['drugs'].to(device)
                labels = batch['labels'].to(device)
                time_indices = batch['time_indices'].to(device)
                
                # 检查批次是否为空
                if vitals.size(0) == 0 or vitals.size(1) == 0:
                    logger.warning(f"批次 {batch_idx} 为空，跳过")
                    continue
                
                # 创建注意力掩码
                # 使用更宽松的掩码条件，考虑多种特征类型
                vitals_valid = vitals.sum(dim=2) != 0  # 至少有一个生命体征特征不为零
                labs_valid = labs.sum(dim=2) != 0      # 至少有一个实验室值特征不为零
                drugs_valid = drugs.sum(dim=2) != 0    # 至少有一个药物特征不为零
                
                # 至少一组特征有效，该位置就视为有效
                valid_positions = vitals_valid | labs_valid | drugs_valid
                
                # 确保特征非零，如果所有特征都是零，添加一些随机特征
                if not valid_positions.any():
                    logger.warning(f"批次 {batch_idx} 中所有特征都是零，添加随机特征")
                    # 添加随机特征到生命体征的第一维
                    batch_size, seq_len = vitals.shape[0], vitals.shape[1]
                    vitals[:, 0, 0] = torch.randn(batch_size, device=device) * 5 + 70  # 模拟心率
                    # 重新计算有效位置
                    vitals_valid = vitals.sum(dim=2) != 0
                    valid_positions = vitals_valid | labs_valid | drugs_valid
                
                # 反转为掩码（True表示填充位置）
                attention_mask = ~valid_positions
                
                # 确保至少有一个非掩码位置
                if attention_mask.all(dim=1).any():
                    # 找出所有位置都被掩码的样本
                    fully_masked = attention_mask.all(dim=1)
                    num_fully_masked = fully_masked.sum().item()
                    if num_fully_masked > 0:
                        logger.warning(f"批次 {batch_idx} 中有{num_fully_masked}个样本的所有位置都被掩码，设置第一个时间点为有效")
                        # 对这些样本，将第一个位置设为非掩码
                        attention_mask[fully_masked, 0] = False
                        # 同时添加随机特征
                        vitals[fully_masked, 0, 0] = torch.randn(num_fully_masked, device=device) * 5 + 70
                        labs[fully_masked, 0, 0] = torch.randn(num_fully_masked, device=device) * 2 + 10
                
                # 为每个样本随机选择一个知识图谱嵌入
                batch_size = vitals.size(0)
                kg_indices = torch.randint(0, len(entity_embeddings), (batch_size,)).to(device)
                kg_embeds = entity_embeddings[kg_indices]
                
                # 简化处理：使用零向量作为文本嵌入
                text_embeds = torch.zeros(batch_size, 768).to(device)
                
                # 前向传播
                outputs = model(vitals, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
                
                # 计算损失（只考虑非填充位置）
                mask = ~attention_mask
                # 检查是否有有效的预测位置
                if not mask.any():
                    logger.warning(f"批次 {batch_idx} 没有有效的预测位置，跳过")
                    continue
                
                masked_outputs = outputs * mask
                masked_labels = labels * mask
                
                # 处理可能的NaN和Inf值
                if torch.isnan(masked_outputs).any() or torch.isinf(masked_outputs).any():
                    logger.warning(f"批次 {batch_idx} 包含NaN或Inf输出值，已替换")
                    masked_outputs = torch.nan_to_num(masked_outputs)
                
                # 确保标签和预测在有效范围内
                masked_outputs = torch.clamp(masked_outputs, 0.0, 1.0)
                masked_labels = torch.clamp(masked_labels, 0.0, 1.0)
                
                loss = criterion(masked_outputs, masked_labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 累计损失
                train_loss += loss.item()
                
                # 收集预测和标签，用于计算AUROC
                train_preds.append(masked_outputs.detach().cpu().numpy())
                train_labels.append(masked_labels.detach().cpu().numpy())
            
            except Exception as e:
                logger.error(f"训练批次 {batch_idx} 处理出错: {e}")
                continue
        
        # 检查是否有有效的训练数据
        if len(train_preds) == 0:
            logger.error("没有有效的训练数据，跳过当前epoch")
            continue
        
        # 计算平均训练损失
        train_loss /= len(train_preds)
        
        # 计算训练集AUROC
        try:
            train_preds = np.concatenate([p.flatten() for p in train_preds])
            train_labels = np.concatenate([l.flatten() for l in train_labels])
            valid_indices = ~np.isnan(train_preds) & ~np.isnan(train_labels)
            
            # 确保有足够的有效数据来计算AUROC
            if np.sum(valid_indices) > 10 and np.sum(train_labels[valid_indices] > 0) > 0 and np.sum(train_labels[valid_indices] == 0) > 0:
                train_auroc = roc_auc_score(train_labels[valid_indices], train_preds[valid_indices])
            else:
                logger.warning("训练集中没有足够的正例和负例来计算AUROC")
                train_auroc = 0.5
        except Exception as e:
            logger.error(f"计算训练集AUROC时出错: {e}")
            train_auroc = 0.5
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                try:
                    # 获取批次数据
                    vitals = batch['vitals'].to(device)
                    labs = batch['labs'].to(device)
                    drugs = batch['drugs'].to(device)
                    labels = batch['labels'].to(device)
                    time_indices = batch['time_indices'].to(device)
                    
                    # 检查批次是否为空
                    if vitals.size(0) == 0 or vitals.size(1) == 0:
                        logger.warning(f"验证批次 {batch_idx} 为空，跳过")
                        continue
                    
                    # 创建注意力掩码
                    # 使用更宽松的掩码条件，考虑多种特征类型
                    vitals_valid = vitals.sum(dim=2) != 0  # 至少有一个生命体征特征不为零
                    labs_valid = labs.sum(dim=2) != 0      # 至少有一个实验室值特征不为零
                    drugs_valid = drugs.sum(dim=2) != 0    # 至少有一个药物特征不为零
                    
                    # 至少一组特征有效，该位置就视为有效
                    valid_positions = vitals_valid | labs_valid | drugs_valid
                    
                    # 确保特征非零，如果所有特征都是零，添加一些随机特征
                    if not valid_positions.any():
                        logger.warning(f"验证批次 {batch_idx} 中所有特征都是零，添加随机特征")
                        # 添加随机特征到生命体征的第一维
                        batch_size, seq_len = vitals.shape[0], vitals.shape[1]
                        vitals[:, 0, 0] = torch.randn(batch_size, device=device) * 5 + 70  # 模拟心率
                        # 重新计算有效位置
                        vitals_valid = vitals.sum(dim=2) != 0
                        valid_positions = vitals_valid | labs_valid | drugs_valid
                    
                    # 反转为掩码（True表示填充位置）
                    attention_mask = ~valid_positions
                    
                    # 确保至少有一个非掩码位置
                    if attention_mask.all(dim=1).any():
                        # 找出所有位置都被掩码的样本
                        fully_masked = attention_mask.all(dim=1)
                        num_fully_masked = fully_masked.sum().item()
                        if num_fully_masked > 0:
                            logger.warning(f"验证批次 {batch_idx} 中有{num_fully_masked}个样本的所有位置都被掩码，设置第一个时间点为有效")
                            # 对这些样本，将第一个位置设为非掩码
                            attention_mask[fully_masked, 0] = False
                            # 同时添加随机特征
                            vitals[fully_masked, 0, 0] = torch.randn(num_fully_masked, device=device) * 5 + 70
                            labs[fully_masked, 0, 0] = torch.randn(num_fully_masked, device=device) * 2 + 10
                    
                    # 为每个样本随机选择一个知识图谱嵌入
                    batch_size = vitals.size(0)
                    kg_indices = torch.randint(0, len(entity_embeddings), (batch_size,)).to(device)
                    kg_embeds = entity_embeddings[kg_indices]
                    
                    # 简化处理：使用零向量作为文本嵌入
                    text_embeds = torch.zeros(batch_size, 768).to(device)
                    
                    # 前向传播
                    outputs = model(vitals, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
                    
                    # 计算损失（只考虑非填充位置）
                    mask = ~attention_mask
                    # 检查是否有有效的预测位置
                    if not mask.any():
                        logger.warning(f"验证批次 {batch_idx} 没有有效的预测位置，跳过")
                        continue
                    
                    masked_outputs = outputs * mask
                    masked_labels = labels * mask
                    
                    # 处理可能的NaN和Inf值
                    if torch.isnan(masked_outputs).any() or torch.isinf(masked_outputs).any():
                        logger.warning(f"验证批次 {batch_idx} 包含NaN或Inf输出值，已替换")
                        masked_outputs = torch.nan_to_num(masked_outputs)
                    
                    # 确保标签和预测在有效范围内
                    masked_outputs = torch.clamp(masked_outputs, 0.0, 1.0)
                    masked_labels = torch.clamp(masked_labels, 0.0, 1.0)
                    
                    loss = criterion(masked_outputs, masked_labels)
                    
                    # 累计损失
                    val_loss += loss.item()
                    
                    # 收集预测和标签，用于计算AUROC
                    val_preds.append(masked_outputs.detach().cpu().numpy())
                    val_labels.append(masked_labels.detach().cpu().numpy())
                
                except Exception as e:
                    logger.error(f"验证批次 {batch_idx} 处理出错: {e}")
                    continue
        
        # 检查是否有有效的验证数据
        if len(val_preds) == 0:
            logger.error("没有有效的验证数据，跳过当前epoch")
            continue
        
        # 计算平均验证损失
        val_loss /= len(val_preds)
        
        # 计算验证集AUROC
        try:
            val_preds = np.concatenate([p.flatten() for p in val_preds])
            val_labels = np.concatenate([l.flatten() for l in val_labels])
            valid_indices = ~np.isnan(val_preds) & ~np.isnan(val_labels)
            
            # 确保有足够的有效数据来计算AUROC
            if np.sum(valid_indices) > 10 and np.sum(val_labels[valid_indices] > 0) > 0 and np.sum(val_labels[valid_indices] == 0) > 0:
                val_auroc = roc_auc_score(val_labels[valid_indices], val_preds[valid_indices])
            else:
                logger.warning("验证集中没有足够的正例和负例来计算AUROC")
                val_auroc = 0.5
        except Exception as e:
            logger.error(f"计算验证集AUROC时出错: {e}")
            val_auroc = 0.5
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(float(train_auroc))
        history['val_auroc'].append(float(val_auroc))
        
        # 计算每个epoch的耗时
        epoch_time = time.time() - start_time
        
        # 输出进度
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Time: {epoch_time:.2f}s - "
                   f"Train Loss: {train_loss:.4f} - "
                   f"Val Loss: {val_loss:.4f} - "
                   f"Train AUROC: {train_auroc:.4f} - "
                   f"Val AUROC: {val_auroc:.4f}")
        
        # 检查早停
        if early_stopping(val_loss, model):
            logger.info(f"早停在epoch {epoch+1}")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 保存训练历史
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # 绘制训练历史
    plot_history(history, output_dir)
    
    # 计算总训练时间
    total_time = time.time() - start_time_total
    logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
    
    return history

def plot_history(history, output_dir):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制AUROC曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_auroc'], 'b-', label='训练AUROC')
    plt.plot(epochs, history['val_auroc'], 'r-', label='验证AUROC')
    plt.title('AUROC曲线')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close() 