# 模型训练和评估的主脚本
def train_and_evaluate_model():
    from utils.data_loading import load_structured_data, preprocess_features
    from utils.dataset import SepsisDataset
    from models.fusion.multimodal_model import SepsisTransformerModel
    from utils.evaluation import evaluate_sepsis_model
    from config import FEATURE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
    
    import torch
    from torch.utils.data import DataLoader, random_split
    import json
    import numpy as np
    import os
    
    # 1. 加载数据
    print("加载处理好的数据...")
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 2. 创建数据集
    print("创建数据集...")
    
    # 使用preprocess_features函数处理特征数据
    vitals, labs, drugs, text_embeds = preprocess_features(patient_features, FEATURE_CONFIG)
    
    # 从sepsis_labels中提取时间索引和目标值
    if 'hour' in sepsis_labels.columns:
        time_indices = sepsis_labels['hour'].values
    elif 'hours_since_admission' in sepsis_labels.columns:
        time_indices = sepsis_labels['hours_since_admission'].values.astype(int)
    else:
        print("警告：在sepsis_labels中找不到时间列，使用默认索引")
        time_indices = np.arange(1, len(sepsis_labels) + 1)
    
    # 重塑为[num_patients, max_seq_len]格式，或创建每个患者的序列
    patient_ids = patient_features['subject_id'].unique()
    max_seq_len = 72  # 默认最大序列长度
    print(f"总数据量: {len(patient_features)}, 患者数量: {len(patient_ids)}")
    
    # 检查text_embeds的维度是否合适，如果过大则降维到合理大小
    if text_embeds.shape[1] > MODEL_CONFIG['text_dim']:
        print(f"文本嵌入维度过大 ({text_embeds.shape[1]}), 将截断至 {MODEL_CONFIG['text_dim']} 维度")
        text_embeds = text_embeds[:, :MODEL_CONFIG['text_dim']]
    
    # 检查kg_embeds的维度是否合适，如果需要调整形状
    # 将kg_embeds调整为[batch_size, num_entities, kg_dim]格式
    if len(kg_embeddings.shape) == 2 and kg_embeddings.shape[0] > 1:
        print(f"将kg_embeds从{kg_embeddings.shape}转换为期望的扩展形式")
        # 创建一个变量来保存扩展后的张量
        kg_embeds_expanded = np.zeros((kg_embeddings.shape[0], 1, kg_embeddings.shape[1]))
        for i in range(kg_embeddings.shape[0]):
            kg_embeds_expanded[i, 0, :] = kg_embeddings[i, :]
        kg_embeddings = kg_embeds_expanded
        print(f"调整后的kg_embeds形状: {kg_embeddings.shape}")
    
    # 初始化结果数组
    reshaped_vitals = np.zeros((len(patient_ids), max_seq_len, vitals.shape[1]))
    reshaped_labs = np.zeros((len(patient_ids), max_seq_len, labs.shape[1]))
    reshaped_drugs = np.zeros((len(patient_ids), max_seq_len, drugs.shape[1]))
    reshaped_time_indices = np.zeros((len(patient_ids), max_seq_len))
    reshaped_targets = np.zeros((len(patient_ids), max_seq_len))
    patient_id_list = []
    
    # 为每个患者创建文本嵌入和知识图谱嵌入
    kg_dim = kg_embeddings.shape[-1]  # 获取最后一个维度
    if len(kg_embeddings.shape) == 3:  # 如果是3D格式 [num_entities, num_concepts, kg_dim]
        print(f"知识图谱嵌入已经是3D格式: {kg_embeddings.shape}")
        kg_embeds_per_patient = np.zeros((len(patient_ids), 1, kg_dim))
    else:  # 如果是2D格式 [num_concepts, kg_dim]
        print(f"知识图谱嵌入是2D格式: {kg_embeddings.shape}")
        kg_embeds_per_patient = np.zeros((len(patient_ids), kg_dim))
    
    # 文本嵌入
    text_embeds_per_patient = np.zeros((len(patient_ids), text_embeds.shape[1]))
    
    print(f"初始化每个患者的文本嵌入维度: {text_embeds_per_patient.shape}")
    print(f"初始化每个患者的知识图谱嵌入维度: {kg_embeds_per_patient.shape}")
    
    # 按患者ID分组，填充数组
    for i, patient_id in enumerate(patient_ids):
        # 获取患者数据
        patient_data_indices = patient_features['subject_id'] == patient_id
        sepsis_data_indices = sepsis_labels['subject_id'] == patient_id
        
        patient_id_list.append(str(patient_id))
        
        # 获取患者的特征数据
        patient_vitals = vitals[patient_data_indices]
        patient_labs = labs[patient_data_indices]
        patient_drugs = drugs[patient_data_indices]
        
        # 获取患者的文本嵌入（取平均值）
        if len(text_embeds[patient_data_indices]) > 0:
            text_embeds_per_patient[i] = np.mean(text_embeds[patient_data_indices], axis=0)
        
        # 获取患者的知识图谱嵌入
        if len(kg_embeddings.shape) == 3:  # 如果是3D格式 [num_entities, num_concepts, kg_dim]
            if len(kg_embeddings) > 0:
                # 只使用第一个实体的嵌入
                kg_embeds_per_patient[i, 0, :] = kg_embeddings[0, 0, :]
        else:  # 如果是2D格式 [num_concepts, kg_dim]
            if len(kg_embeddings) > 0:
                kg_embeds_per_patient[i] = kg_embeddings[0]
        
        # 获取时间索引
        patient_times = time_indices[sepsis_data_indices]
        seq_len = min(len(patient_times), max_seq_len, len(patient_vitals))
        
        if seq_len > 0:
            # 填充特征数据
            reshaped_vitals[i, :seq_len] = patient_vitals[:seq_len]
            reshaped_labs[i, :seq_len] = patient_labs[:seq_len]
            reshaped_drugs[i, :seq_len] = patient_drugs[:seq_len]
            reshaped_time_indices[i, :seq_len] = np.arange(1, seq_len + 1)  # 从1开始的序列索引
            
            # 获取目标值
            if 'sepsis_prediction_window' in sepsis_labels.columns:
                patient_targets = sepsis_labels.loc[sepsis_data_indices, 'sepsis_prediction_window'].values
            elif 'sepsis3' in sepsis_labels.columns:
                patient_targets = sepsis_labels.loc[sepsis_data_indices, 'sepsis3'].values
            elif 'sepsis_label' in sepsis_labels.columns:
                patient_targets = sepsis_labels.loc[sepsis_data_indices, 'sepsis_label'].values
            else:
                print(f"警告：患者 {patient_id} 在sepsis_labels中找不到目标列，使用全零目标")
                patient_targets = np.zeros(len(patient_times))
            
            # 填充目标值
            reshaped_targets[i, :seq_len] = patient_targets[:seq_len]
    
    # 使用处理后的数据
    vitals = reshaped_vitals
    labs = reshaped_labs
    drugs = reshaped_drugs

    # 将text_embeds从2D(batch_size, embed_dim)扩展为3D(batch_size, seq_len, embed_dim)
    # 每个时间步使用相同的文本嵌入
    text_embeds_expanded = np.zeros((len(patient_ids), max_seq_len, text_embeds_per_patient.shape[1]))
    for i in range(len(patient_ids)):
        # 对于每个患者，将其文本嵌入复制到所有时间步
        for t in range(max_seq_len):
            text_embeds_expanded[i, t, :] = text_embeds_per_patient[i, :]

    # 使用扩展后的文本嵌入
    text_embeds = text_embeds_expanded
    print(f"- text_embeds (扩展后): {text_embeds.shape}")

    # 使用处理后的其他数据
    kg_embeds = kg_embeds_per_patient
    time_indices = reshaped_time_indices
    targets = reshaped_targets
    patient_ids = patient_id_list
    
    print(f"处理后的数据形状:") 
    print(f"- vitals: {vitals.shape}")
    print(f"- labs: {labs.shape}")
    print(f"- drugs: {drugs.shape}")
    print(f"- text_embeds: {text_embeds.shape}")
    print(f"- kg_embeds: {kg_embeds.shape}")
    print(f"- time_indices: {time_indices.shape}")
    print(f"- targets: {targets.shape}")
    print(f"- 患者ID数: {len(patient_ids)}")
    
    # 确保目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 创建SepsisDataset实例，按照其正确的参数格式
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
        use_augmentation=TRAIN_CONFIG.get('use_data_augmentation', True)
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
    
    # 7. 定义损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # 8. 训练循环
    print("开始训练模型...")
    num_epochs = TRAIN_CONFIG['num_epochs']
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

if __name__ == "__main__":
    train_and_evaluate_model()