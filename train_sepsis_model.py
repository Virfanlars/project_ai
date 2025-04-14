import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# 导入自定义模块
from utils.data_loading import load_structured_data
from utils.dataset import SepsisDataset
from models.fusion.multimodal_model import SepsisTransformerModel
from utils.evaluation import evaluate_sepsis_model
from config import MODEL_CONFIG, TRAIN_CONFIG, FEATURE_CONFIG

def main():
    # 1. 加载数据
    print("加载数据...")
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 2. 创建数据集
    print("创建数据集...")
    dataset = SepsisDataset(
        patient_features=patient_features,
        sepsis_labels=sepsis_labels,
        kg_embeddings=kg_embeddings,
        time_axis=time_axis,
        feature_config=FEATURE_CONFIG,
        max_seq_len=48  # 假设最长序列为2天
    )
    
    # 3. 数据集划分
    print("划分数据集...")
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 5. 创建模型
    print("创建模型...")
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
    
    # 6. 训练模型
    print("开始训练...")
    device = TRAIN_CONFIG['device']
    model = model.to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # 早停设置
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")
        for batch in progress_bar:
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
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                    item.to(device) if torch.is_tensor(item) else item for item in batch
                ]
                
                outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pt')
            print("模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 7. 评估模型
    print("加载最佳模型进行评估...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_results = evaluate_sepsis_model(model, test_loader, device=device)
    
    print("\n评估结果:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # 8. 保存评估结果
    with open('evaluation_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print("评估结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    main() 