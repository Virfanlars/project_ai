#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建数据集划分信息文件
"""

import os
import torch
import numpy as np

def create_dataset_splits():
    """创建并保存数据集划分信息"""
    print("创建数据集划分信息文件...")
    
    # 假设有1000个样本
    n_samples = 1000
    
    # 创建随机索引
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 划分训练集、验证集和测试集
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 转换为PyTorch张量
    train_indices = torch.LongTensor(train_indices)
    val_indices = torch.LongTensor(val_indices)
    test_indices = torch.LongTensor(test_indices)
    
    # 保存数据集划分信息
    os.makedirs('models', exist_ok=True)
    
    # 兼容Windows和Linux路径
    splits_path = 'models\\dataset_splits.pt'
    
    # 保存数据
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
    }, splits_path)
    
    print(f"数据集划分信息已保存到{splits_path}")
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    print(f"测试集大小: {len(test_indices)}")

if __name__ == "__main__":
    create_dataset_splits() 