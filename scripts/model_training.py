# 模型训练和评估的主脚本
def train_and_evaluate_model():
    from utils.data_loading import load_structured_data, preprocess_features
    from utils.dataset import SepsisDataset
    from models.fusion.multimodal_model import SepsisTransformerModel
    from utils.evaluation import evaluate_sepsis_model
    from config import FEATURE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    import json
    import numpy as np
    import os
    
    # 早停类定义
    class EarlyStopping:
        def __init__(self, patience=7, verbose=True, delta=0, path='models/best_model.pt'):
            """
            参数:
                patience (int): 验证集损失多少个epoch不下降就停止训练
                verbose (bool): 是否打印早停信息
                delta (float): 认为是改进的最小变化
                path (str): 模型保存路径
            """
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.delta = delta
            self.path = path

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping 计数器: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            '''验证损失decreased时保存模型'''
            if self.verbose:
                print(f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
    
    # 加载训练配置
    batch_size = TRAIN_CONFIG.get('batch_size', 64)
    learning_rate = TRAIN_CONFIG.get('learning_rate', 0.001)
    num_epochs = TRAIN_CONFIG.get('num_epochs', 50)
    early_stopping_patience = TRAIN_CONFIG.get('early_stopping_patience', 10)
    weight_decay = TRAIN_CONFIG.get('weight_decay', 1e-4)
    
    # 打印配置信息
    print(f"\n训练配置: 批量大小={batch_size}, 学习率={learning_rate}, 训练轮数={num_epochs}")
    print(f"模型配置: hidden_dim={MODEL_CONFIG['hidden_dim']}, heads={MODEL_CONFIG['num_heads']}, layers={MODEL_CONFIG['num_layers']}")
    
    # 获取目标列配置
    target_column = DATA_CONFIG.get('target_column', 'sepsis_label')
    print(f"使用目标列: {target_column}")
    
    # 6. 根据配置加载数据
    print("加载数据...")
    vitals, labs, drugs, text_embeds, kg_embeds, time_indices, targets, patient_ids = load_data(
        target_column=target_column  # 明确传递目标列参数
    )
    
    # 检查标签分布
    print('训练标签分布:', np.unique(targets, return_counts=True))
    
    # 打印特征统计信息
    print('特征均值/标准差:')
    print('vitals 均值:', np.nanmean(vitals), '方差:', np.nanvar(vitals))
    print('labs 均值:', np.nanmean(labs), '方差:', np.nanvar(labs))
    print('drugs 均值:', np.nanmean(drugs), '方差:', np.nanvar(drugs))
    print('text_embeds 均值:', np.nanmean(text_embeds), '方差:', np.nanvar(text_embeds))
    print('kg_embeds 均值:', np.nanmean(kg_embeds), '方差:', np.nanvar(kg_embeds))
    
    # 7. 创建数据集和数据加载器
    # 创建SepsisDataset实例
    dataset = SepsisDataset(
        vitals=vitals,
        labs=labs,
        drugs=drugs,
        text_embeds=text_embeds,
        kg_embeds=kg_embeds,
        time_indices=time_indices,
        targets=targets,
        patient_ids=patient_ids,
        max_seq_len=MODEL_CONFIG.get('max_seq_len', 72),
        use_augmentation=TRAIN_CONFIG.get('use_data_augmentation', False)  # 关闭数据增强简化调试
    )
    
    # 3. 数据集划分
    print("划分训练、验证和测试集...")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 自定义collate_fn，处理None值
    def custom_collate_fn(batch):
        # 过滤掉包含None的样本
        filtered_batch = []
        for item in batch:
            if None not in item:
                filtered_batch.append(item)
        
        # 如果过滤后的批次为空，返回一个占位样本
        if not filtered_batch:
            # 创建一个有效的占位样本，确保形状正确
            max_seq_len = 48  # 最大序列长度
            vitals = torch.zeros(1, max_seq_len, MODEL_CONFIG['vitals_dim'])
            labs = torch.zeros(1, max_seq_len, MODEL_CONFIG['lab_dim'])
            drugs = torch.zeros(1, max_seq_len, MODEL_CONFIG['drug_dim'])
            text_embed = torch.zeros(1, max_seq_len, MODEL_CONFIG['text_dim'])
            kg_embed = torch.zeros(1, MODEL_CONFIG['kg_dim'])
            time_indices = torch.zeros(1, max_seq_len, dtype=torch.long)
            labels = torch.zeros(1, max_seq_len)
            return [vitals, labs, drugs, text_embed, kg_embed, time_indices, labels, None]
        
        # 对于非空批次，按正常方式处理
        vitals = torch.stack([x[0] for x in filtered_batch])
        labs = torch.stack([x[1] for x in filtered_batch])
        drugs = torch.stack([x[2] for x in filtered_batch])
        text_embed = torch.stack([x[3] for x in filtered_batch])
        
        # 知识图谱嵌入是一个向量，而不是序列
        # 每个样本只有一个KG嵌入向量，形状为[kg_dim]
        kg_embed = torch.stack([x[4] for x in filtered_batch])  # 形状为[batch_size, kg_dim]
        
        time_indices = torch.stack([x[5] for x in filtered_batch])
        labels = torch.stack([x[6] for x in filtered_batch])
        
        # onset_time是可选的，可能为None
        onset_times = [x[7] for x in filtered_batch]
        
        return [vitals, labs, drugs, text_embed, kg_embed, time_indices, labels, onset_times]
    
    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # 5. 初始化模型
    print("初始化模型...")
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
    
    # 6. 设置设备
    device = torch.device(TRAIN_CONFIG['device'])
    model.to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 计算类别权重
    class_weights = TRAIN_CONFIG.get('class_weights', [0.5, 0.5])
    weight_tensor = torch.tensor(class_weights, device=device)
    criterion = nn.BCELoss(weight=None)  # 先不使用权重，观察基本性能
    
    print(f"类别权重: {class_weights}")
    
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=TRAIN_CONFIG.get('lr_scheduler_factor', 0.5), 
        patience=TRAIN_CONFIG.get('lr_scheduler_patience', 5), verbose=True
    )
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, verbose=True, path='models/best_model.pt'
    )
    
    # 8. 训练循环
    print("开始训练模型...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                item.to(device) if torch.is_tensor(item) else item for item in batch
            ]
            
            optimizer.zero_grad()
            outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
            
            # 确保outputs和targets形状匹配
            # outputs形状: [batch_size, seq_len, 1]
            # 需要调整为: [batch_size*seq_len]
            outputs = outputs.view(-1)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, _ = [
                    item.to(device) if torch.is_tensor(item) else item for item in batch
                ]
                
                outputs = model(vitals, labs, drugs, text_embed, kg_embed, time_indices)
                
                # 同样调整验证时的形状
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models\\best_model.pt')
            print("模型已保存到 models\\best_model.pt")
    
    # 9. 评估模型
    print("评估模型性能...")
    model.load_state_dict(torch.load('models\\best_model.pt'))
    test_results = evaluate_sepsis_model(model, test_loader, device)
    
    print("\n测试集评估结果:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存评估结果
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # 保存数据集划分信息，用于后续分析
    torch.save({
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
    }, 'models\\dataset_splits.pt')
    print("数据集划分信息已保存到 models\\dataset_splits.pt")
    
    return model, test_dataset 

def load_data(target_column='sepsis_label'):
    """
    加载数据，准备模型训练所需的特征和标签
    
    参数:
        target_column: 目标列名，默认使用'sepsis_label'
        
    返回:
        模型训练所需的各类特征和标签
    """
    # 在函数内部导入所需模块
    from utils.data_loading import load_structured_data, preprocess_features
    from config import FEATURE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
    import numpy as np
    
    # 1. 加载数据
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 确认目标列存在
    if target_column not in sepsis_labels.columns:
        print(f"警告: 目标列 '{target_column}' 不在sepsis_labels中。可用列: {sepsis_labels.columns}")
        if 'sepsis_label' in sepsis_labels.columns:
            print(f"使用'sepsis_label'作为替代目标")
            target_column = 'sepsis_label'
        elif 'sepsis3' in sepsis_labels.columns:
            print(f"使用'sepsis3'作为替代目标")
            target_column = 'sepsis3'
        else:
            raise ValueError(f"找不到合适的目标列")
    
    # 2. 提取特征
    vitals, labs, drugs, text_embeds = preprocess_features(patient_features, FEATURE_CONFIG)
    
    # 3. 获取唯一患者ID
    patient_ids = patient_features['subject_id'].unique()
    
    # 4. 设置最大序列长度
    max_seq_len = DATA_CONFIG.get('max_seq_len', 72)
    
    # 5. 初始化数据结构
    n_patients = len(patient_ids)
    n_vital_features = vitals.shape[1]
    n_lab_features = labs.shape[1]
    n_drug_features = drugs.shape[1]
    n_text_features = text_embeds.shape[1]
    
    reshaped_vitals = np.zeros((n_patients, max_seq_len, n_vital_features))
    reshaped_labs = np.zeros((n_patients, max_seq_len, n_lab_features))
    reshaped_drugs = np.zeros((n_patients, max_seq_len, n_drug_features))
    reshaped_text_embeds = np.zeros((n_patients, max_seq_len, n_text_features))
    reshaped_time_indices = np.zeros((n_patients, max_seq_len), dtype=int)
    reshaped_targets = np.zeros((n_patients, max_seq_len))
    
    # 6. 处理每个患者的数据
    print(f"处理{n_patients}名患者的数据...")
    
    for i, patient_id in enumerate(patient_ids):
        if i % 500 == 0:
            print(f"处理患者 {i}/{n_patients}...")
        
        # 获取患者数据
        patient_data_indices = patient_features['subject_id'] == patient_id
        patient_vitals = vitals[patient_data_indices]
        patient_labs = labs[patient_data_indices]
        patient_drugs = drugs[patient_data_indices]
        patient_text_embeds = text_embeds[patient_data_indices]
        
        # 获取时间信息
        sepsis_data_indices = sepsis_labels['subject_id'] == patient_id
        
        # 确保有匹配的标签数据
        if sepsis_data_indices.sum() == 0:
            print(f"警告：患者 {patient_id} 在sepsis_labels中没有匹配记录")
            continue
            
        # 根据hour列获取时间索引
        if 'hour' in sepsis_labels.columns:
            patient_times = sepsis_labels.loc[sepsis_data_indices, 'hour'].values
        else:
            patient_times = np.arange(len(patient_vitals))
            
        # 获取目标值 - 使用指定的目标列
        if target_column in sepsis_labels.columns:
            patient_targets = sepsis_labels.loc[sepsis_data_indices, target_column].values
            print(f"患者 {patient_id} 目标分布: {np.unique(patient_targets, return_counts=True)}")
        else:
            print(f"警告：患者 {patient_id} 在sepsis_labels中找不到目标列 {target_column}，使用全零目标")
            patient_targets = np.zeros(len(patient_times))
        
        # 截断序列长度
        seq_len = min(len(patient_times), max_seq_len, len(patient_vitals))
        if seq_len > 0:
            # 填充数据
            reshaped_vitals[i, :seq_len] = patient_vitals[:seq_len]
            reshaped_labs[i, :seq_len] = patient_labs[:seq_len]
            reshaped_drugs[i, :seq_len] = patient_drugs[:seq_len]
            reshaped_text_embeds[i, :seq_len] = patient_text_embeds[:seq_len]
            reshaped_time_indices[i, :seq_len] = np.arange(1, seq_len+1)  # 时间索引从1开始
            reshaped_targets[i, :seq_len] = patient_targets[:seq_len]
    
    return reshaped_vitals, reshaped_labs, reshaped_drugs, reshaped_text_embeds, kg_embeddings, reshaped_time_indices, reshaped_targets, patient_ids

if __name__ == "__main__":
    train_and_evaluate_model()