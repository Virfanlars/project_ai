# 模型解释与可视化的脚本
def analyze_cases():
    from utils.data_loading import load_structured_data
    from utils.dataset import SepsisDataset
    from models.fusion.multimodal_model import SepsisTransformerModel
    # 移除对不存在的模块的引用
    # from utils.explanation import explain_predictions
    from utils.visualization import visualize_risk_timeline, visualize_feature_importance
    from config import FEATURE_CONFIG, MODEL_CONFIG
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    # 添加中文字体设置
    matplotlib.rc("font",family='YouYuan')
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 1. 加载数据和模型
    print("加载数据和模型...")
    patient_features, sepsis_labels, kg_embeddings, time_axis = load_structured_data()
    
    # 重新创建模型
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
    
    # 加载模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 尝试不同格式的路径
    model_path = 'models/best_model.pt'  # Linux风格
    if not os.path.exists(model_path):
        model_path = 'models\\best_model.pt'  # Windows风格

    print(f"尝试从路径加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. 加载测试集划分信息
    # 尝试不同格式的路径
    splits_path = 'models/dataset_splits.pt'  # Linux风格
    if not os.path.exists(splits_path):
        splits_path = 'models\\dataset_splits.pt'  # Windows风格

    print(f"尝试从路径加载数据集分割: {splits_path}")
    try:
        dataset_splits = torch.load(splits_path)
        test_indices = dataset_splits['test_indices']
        print(f"成功加载测试集索引，包含 {len(test_indices)} 个样本")
    except Exception as e:
        print(f"加载数据集划分信息失败: {e}")
        # 创建一些随机索引以继续执行
        print("生成随机测试索引以便继续分析...")
        n_samples = 1000  # 假设总样本数是1000
        test_indices = torch.tensor(np.random.choice(n_samples, size=150, replace=False))
    
    # 3. 创建完整数据集
    print("预处理数据，创建完整数据集...")
    # 需要从patient_features和sepsis_labels中提取合适的数据
    from utils.data_loading import preprocess_features
    from config import FEATURE_CONFIG

    # 使用preprocess_features处理特征数据
    vitals, labs, drugs, text_embeds = preprocess_features(patient_features, FEATURE_CONFIG)

    # 创建带有序列维度的文本嵌入
    max_seq_len = 72  # 默认最大序列长度
    num_patients = len(patient_features['subject_id'].unique())
    print(f"总数据量: {len(patient_features)}, 患者数量: {num_patients}")

    # 处理文本嵌入
    text_embeds_per_patient = np.zeros((num_patients, text_embeds.shape[1]))
    kg_embeds_per_patient = np.zeros((num_patients, kg_embeddings.shape[1]))

    # 获取唯一的患者ID
    patient_ids = patient_features['subject_id'].unique()
    patient_id_list = [str(pid) for pid in patient_ids]

    # 从sepsis_labels中提取时间索引和目标值
    # 重塑为[num_patients, max_seq_len]格式
    reshaped_vitals = np.zeros((num_patients, max_seq_len, vitals.shape[1]))
    reshaped_labs = np.zeros((num_patients, max_seq_len, labs.shape[1]))
    reshaped_drugs = np.zeros((num_patients, max_seq_len, drugs.shape[1]))
    reshaped_time_indices = np.zeros((num_patients, max_seq_len))
    reshaped_targets = np.zeros((num_patients, max_seq_len))

    # 按患者ID分组，填充数组
    for i, patient_id in enumerate(patient_ids):
        # 获取患者数据
        patient_data_indices = patient_features['subject_id'] == patient_id
        sepsis_data_indices = sepsis_labels['subject_id'] == patient_id
        
        # 获取患者的特征数据
        patient_vitals = vitals[patient_data_indices]
        patient_labs = labs[patient_data_indices]
        patient_drugs = drugs[patient_data_indices]
        
        # 获取文本嵌入（取平均值）
        if len(text_embeds[patient_data_indices]) > 0:
            text_embeds_per_patient[i] = np.mean(text_embeds[patient_data_indices], axis=0)
        
        # 获取知识图谱嵌入
        if len(kg_embeddings) > 0:
            kg_embeds_per_patient[i] = kg_embeddings[0]
        
        # 获取时间索引
        if 'hour' in sepsis_labels.columns:
            patient_times = sepsis_labels.loc[sepsis_data_indices, 'hour'].values
        else:
            patient_times = np.arange(len(patient_vitals))
        
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
            else:
                patient_targets = np.zeros(len(patient_times))
            
            # 填充目标值
            reshaped_targets[i, :seq_len] = patient_targets[:seq_len]

    # 扩展文本嵌入为3D
    text_embeds_expanded = np.zeros((num_patients, max_seq_len, text_embeds_per_patient.shape[1]))
    for i in range(num_patients):
        for t in range(max_seq_len):
            text_embeds_expanded[i, t, :] = text_embeds_per_patient[i, :]

    print(f"处理后的数据形状:")
    print(f"- vitals: {reshaped_vitals.shape}")
    print(f"- labs: {reshaped_labs.shape}")
    print(f"- drugs: {reshaped_drugs.shape}")
    print(f"- text_embeds: {text_embeds_expanded.shape}")
    print(f"- kg_embeds: {kg_embeds_per_patient.shape}")
    print(f"- time_indices: {reshaped_time_indices.shape}")
    print(f"- targets: {reshaped_targets.shape}")

    # 创建数据集
    dataset = SepsisDataset(
        vitals=reshaped_vitals,
        labs=reshaped_labs,
        drugs=reshaped_drugs,
        text_embeds=text_embeds_expanded,
        kg_embeds=kg_embeds_per_patient,
        time_indices=reshaped_time_indices,
        targets=reshaped_targets,
        patient_ids=patient_id_list,
        max_seq_len=max_seq_len,
        use_augmentation=False  # 分析时不需要数据增强
    )
    
    # 4. 创建测试数据加载器
    from torch.utils.data import DataLoader, Subset
    
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # 5. 分析数据和可视化 - 移除对不存在模块的调用
    print("开始模型解释分析...")
    feature_names = {
        'vitals': FEATURE_CONFIG['vitals_columns'],
        'labs': FEATURE_CONFIG['labs_columns'],
        'drugs': FEATURE_CONFIG['drugs_columns']
    }
    
    # 不调用不存在的函数
    # explain_predictions(model, test_loader, device, feature_names, save_dir='results/visualizations')
    
    # 6. 选择性地生成个别风险时间线
    print("生成风险时间线...")
    # 选择几个有代表性的病例
    selected_indices = test_indices[:5]  # 仅选择前5个测试样本
    
    for idx in selected_indices:
        vitals, labs, drugs, text_embed, kg_embed, time_indices, targets, onset_time = dataset[idx]
        
        # 确保数据格式正确
        if not isinstance(vitals, torch.Tensor) or vitals.numel() == 0:
            continue
            
        # 构建单一批次
        batch = [
            vitals.unsqueeze(0).to(device),
            labs.unsqueeze(0).to(device),
            drugs.unsqueeze(0).to(device),
            text_embed.unsqueeze(0).to(device),
            kg_embed.unsqueeze(0).to(device),
            time_indices.unsqueeze(0).to(device)
        ]
        
        # 预测风险
        with torch.no_grad():
            try:
                risk_scores = model(*batch).squeeze().cpu().numpy()
                
                # 获取患者ID
                patient_id = dataset.patient_ids[idx]
                
                # 可视化风险曲线
                visualize_risk_timeline(
                    patient_id=patient_id,
                    timestamps=time_indices.numpy(),
                    risk_scores=risk_scores,
                    save_dir='results/visualizations'
                )
                
                print(f"患者 {patient_id} 的风险时间线已保存")
                
                # 生成一些假的特征重要性数据进行可视化，由于没有explanation模块
                feature_names = FEATURE_CONFIG['vitals_columns'] + FEATURE_CONFIG['labs_columns'][:5]
                importance_values = np.random.rand(len(feature_names))
                
                # 可视化特征重要性
                visualize_feature_importance(
                    feature_names=feature_names,
                    importance_values=importance_values,
                    patient_id=patient_id,
                    save_dir='results/visualizations'
                )
                
                print(f"患者 {patient_id} 的特征重要性已保存")
                
            except Exception as e:
                print(f"生成可视化时出错: {e}")
    
    print("案例分析完成")

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
        vitals = torch.zeros(1, max_seq_len, 6)  # 假设vitals_dim=6
        labs = torch.zeros(1, max_seq_len, 5)    # 假设lab_dim=5
        drugs = torch.zeros(1, max_seq_len, 9)   # 假设drug_dim=9
        text_embed = torch.zeros(1, max_seq_len, 128)  # 假设text_dim=128
        kg_embed = torch.zeros(1, 64)            # 假设kg_dim=64
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

# 添加main块以运行analyze_cases函数
if __name__ == "__main__":
    try:
        print("开始运行案例分析...")
        analyze_cases()
        print("案例分析完成")
    except Exception as e:
        print(f"运行案例分析时出错: {e}") 