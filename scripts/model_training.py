# 模型训练和评估的主脚本
def train_and_evaluate_model():
    from utils.data_loading import load_structured_data
    from utils.dataset import SepsisDataset
    from models.fusion.multimodal_model import SepsisTransformerModel
    from utils.evaluation import evaluate_sepsis_model
    from config import FEATURE_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
    
    import torch
    from torch.utils.data import DataLoader, random_split
    import json
    
    # 1. 加载数据
    print("加载处理好的数据...")
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 2. 创建数据集
    print("创建数据集...")
    dataset = SepsisDataset(
        patient_features=patient_features,
        sepsis_labels=sepsis_labels,
        kg_embeddings=kg_embeddings,
        time_axis=time_axis,
        feature_config=FEATURE_CONFIG
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
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("模型已保存")
    
    # 9. 评估模型
    print("评估模型性能...")
    model.load_state_dict(torch.load('models/best_model.pt'))
    test_results = evaluate_sepsis_model(model, test_loader, device)
    
    print("\n测试集评估结果:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存评估结果
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # 保存数据集划分信息，用于后续分析
    torch.save({
        'test_indices': test_dataset.indices,
    }, 'models/dataset_splits.pt')
    
    return model, test_dataset 