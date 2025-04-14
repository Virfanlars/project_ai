# 数据集定义
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.data_loading import preprocess_features
from sklearn.preprocessing import StandardScaler
import os
import random
from scipy.interpolate import interp1d

class SepsisDataset(Dataset):
    """
    脓毒症预测数据集类
    
    支持时间序列数据、文本和图嵌入的多模态输入
    实现数据增强和时间扰动，显著扩大数据集规模
    """
    def __init__(self, vitals, labs, drugs, text_embeds, kg_embeds, time_indices, 
                 targets, patient_ids, max_seq_len=72, use_augmentation=True,
                 time_shift_prob=0.5, value_perturb_prob=0.5, perturb_scale=0.1):
        """
        初始化数据集
        
        参数:
            vitals: 生命体征数据 [n_samples, seq_len, vitals_dim]
            labs: 实验室检查数据 [n_samples, seq_len, lab_dim]
            drugs: 药物数据 [n_samples, seq_len, drug_dim]
            text_embeds: 文本嵌入 [n_samples, text_dim]
            kg_embeds: 知识图谱嵌入 [n_samples, kg_dim]
            time_indices: 时间索引 [n_samples, seq_len]
            targets: 目标值 [n_samples, seq_len]
            patient_ids: 患者ID列表
            max_seq_len: 最大序列长度
            use_augmentation: 是否使用数据增强
            time_shift_prob: 时间偏移概率
            value_perturb_prob: 数值扰动概率
            perturb_scale: 扰动尺度
        """
        self.vitals = vitals
        self.labs = labs
        self.drugs = drugs
        self.text_embeds = text_embeds
        self.kg_embeds = kg_embeds
        self.time_indices = time_indices
        self.targets = targets
        self.patient_ids = patient_ids
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation
        self.time_shift_prob = time_shift_prob
        self.value_perturb_prob = value_perturb_prob
        self.perturb_scale = perturb_scale
        
        # 确保所有输入具有相同的样本数
        assert len(vitals) == len(labs) == len(drugs) == len(text_embeds) == len(kg_embeds) == len(time_indices) == len(targets)
        
        # 如果启用增强，预先生成一些增强样本
        if use_augmentation:
            print("启用数据增强，生成扩展数据集...")
            self._augment_dataset()
        
    def __len__(self):
        return len(self.vitals)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        vitals = torch.tensor(self.vitals[idx], dtype=torch.float32)
        labs = torch.tensor(self.labs[idx], dtype=torch.float32)
        drugs = torch.tensor(self.drugs[idx], dtype=torch.float32)
        text_embed = torch.tensor(self.text_embeds[idx], dtype=torch.float32)
        kg_embed = torch.tensor(self.kg_embeds[idx], dtype=torch.float32)
        time_indices = torch.tensor(self.time_indices[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        patient_id = self.patient_ids[idx]
        
        return vitals, labs, drugs, text_embed, kg_embed, time_indices, target, patient_id
    
    def _augment_dataset(self):
        """
        通过数据增强扩展数据集
        
        包括:
        1. 时间偏移 - 随机移动序列起点
        2. 值扰动 - 添加随机噪声到连续特征
        """
        n_original = len(self.vitals)
        n_augmented = n_original  # 增强后的总样本数将是原始数量的2倍
        
        # 为增强样本预分配内存
        aug_vitals = np.zeros((n_augmented, self.max_seq_len, self.vitals.shape[2]), dtype=np.float32)
        aug_labs = np.zeros((n_augmented, self.max_seq_len, self.labs.shape[2]), dtype=np.float32)
        aug_drugs = np.zeros((n_augmented, self.max_seq_len, self.drugs.shape[2]), dtype=np.float32)
        aug_time_indices = np.zeros((n_augmented, self.max_seq_len), dtype=np.int64)
        aug_targets = np.zeros((n_augmented, self.max_seq_len), dtype=np.float32)
        
        # 复制文本和知识图嵌入 (这些不进行时序增强)
        aug_text_embeds = np.copy(self.text_embeds)
        aug_kg_embeds = np.copy(self.kg_embeds)
        aug_patient_ids = self.patient_ids.copy()
        
        print(f"正在生成 {n_augmented} 个增强样本...")
        for i in range(n_augmented):
            # 选择一个原始样本进行增强
            orig_idx = i % n_original
            
            # 应用时间偏移增强
            if random.random() < self.time_shift_prob:
                self._apply_time_shift(
                    orig_idx, i,
                    aug_vitals, aug_labs, aug_drugs, aug_time_indices, aug_targets
                )
            # 应用值扰动增强
            elif random.random() < self.value_perturb_prob:
                self._apply_value_perturbation(
                    orig_idx, i,
                    aug_vitals, aug_labs, aug_drugs, aug_time_indices, aug_targets
                )
            # 如果都不应用，直接复制原始样本
            else:
                aug_vitals[i] = self.vitals[orig_idx]
                aug_labs[i] = self.labs[orig_idx]
                aug_drugs[i] = self.drugs[orig_idx]
                aug_time_indices[i] = self.time_indices[orig_idx]
                aug_targets[i] = self.targets[orig_idx]
        
        # 合并原始数据和增强数据
        self.vitals = np.concatenate([self.vitals, aug_vitals], axis=0)
        self.labs = np.concatenate([self.labs, aug_labs], axis=0)
        self.drugs = np.concatenate([self.drugs, aug_drugs], axis=0)
        self.text_embeds = np.concatenate([self.text_embeds, aug_text_embeds], axis=0)
        self.kg_embeds = np.concatenate([self.kg_embeds, aug_kg_embeds], axis=0)
        self.time_indices = np.concatenate([self.time_indices, aug_time_indices], axis=0)
        self.targets = np.concatenate([self.targets, aug_targets], axis=0)
        
        # 更新患者ID，为增强样本添加后缀
        aug_patient_ids = [f"{pid}_aug" for pid in aug_patient_ids]
        self.patient_ids.extend(aug_patient_ids)
        
        print(f"增强后的数据集大小: {len(self.vitals)} 样本")
    
    def _apply_time_shift(self, orig_idx, aug_idx, aug_vitals, aug_labs, aug_drugs, aug_time_indices, aug_targets):
        """
        应用时间偏移增强
        
        参数:
            orig_idx: 原始样本索引
            aug_idx: 增强样本索引
            aug_*: 用于存储增强样本的数组
        """
        # 确定原始样本中有效的时间点数量
        valid_time_mask = self.time_indices[orig_idx] > 0
        n_valid_times = np.sum(valid_time_mask)
        
        if n_valid_times <= 1:
            # 如果只有0或1个有效时间点，无法进行偏移，直接复制
            aug_vitals[aug_idx] = self.vitals[orig_idx]
            aug_labs[aug_idx] = self.labs[orig_idx]
            aug_drugs[aug_idx] = self.drugs[orig_idx]
            aug_time_indices[aug_idx] = self.time_indices[orig_idx]
            aug_targets[aug_idx] = self.targets[orig_idx]
            return
        
        # 随机选择偏移量，范围是1到有效时间点数量的一半
        max_shift = max(1, n_valid_times // 2)
        shift = random.randint(1, max_shift)
        
        # 应用偏移 - 删除前shift个时间点
        aug_vitals[aug_idx, :n_valid_times-shift] = self.vitals[orig_idx, shift:n_valid_times]
        aug_labs[aug_idx, :n_valid_times-shift] = self.labs[orig_idx, shift:n_valid_times]
        aug_drugs[aug_idx, :n_valid_times-shift] = self.drugs[orig_idx, shift:n_valid_times]
        
        # 调整时间索引
        new_time_indices = np.zeros(self.max_seq_len, dtype=np.int64)
        new_time_indices[:n_valid_times-shift] = np.arange(1, n_valid_times-shift+1)
        aug_time_indices[aug_idx] = new_time_indices
        
        # 调整目标值
        aug_targets[aug_idx, :n_valid_times-shift] = self.targets[orig_idx, shift:n_valid_times]
    
    def _apply_value_perturbation(self, orig_idx, aug_idx, aug_vitals, aug_labs, aug_drugs, aug_time_indices, aug_targets):
        """
        应用值扰动增强
        
        参数:
            orig_idx: 原始样本索引
            aug_idx: 增强样本索引
            aug_*: 用于存储增强样本的数组
        """
        # 复制原始时间索引和目标值 (这些不变)
        aug_time_indices[aug_idx] = self.time_indices[orig_idx]
        aug_targets[aug_idx] = self.targets[orig_idx]
        
        # 确定原始样本中有效的时间点数量
        valid_time_mask = self.time_indices[orig_idx] > 0
        n_valid_times = np.sum(valid_time_mask)
        
        # 为生命体征添加随机噪声
        vitals_noise = np.random.normal(0, self.perturb_scale, 
                                      (n_valid_times, self.vitals.shape[2]))
        aug_vitals[aug_idx, :n_valid_times] = self.vitals[orig_idx, :n_valid_times] * (1 + vitals_noise)
        
        # 为实验室值添加随机噪声
        labs_noise = np.random.normal(0, self.perturb_scale, 
                                    (n_valid_times, self.labs.shape[2]))
        aug_labs[aug_idx, :n_valid_times] = self.labs[orig_idx, :n_valid_times] * (1 + labs_noise)
        
        # 药物使用通常是二元的，我们可以随机翻转一些值
        aug_drugs[aug_idx] = self.drugs[orig_idx].copy()
        if n_valid_times > 0 and self.drugs.shape[2] > 0:
            # 对每个有效时间点，有10%的几率翻转一个药物值
            for t in range(n_valid_times):
                if random.random() < 0.1 and np.sum(self.drugs[orig_idx, t]) > 0:
                    drug_idx = random.randint(0, self.drugs.shape[2]-1)
                    aug_drugs[aug_idx, t, drug_idx] = 1 - aug_drugs[aug_idx, t, drug_idx]

def create_datasets(data_path, split_ratio=[0.7, 0.15, 0.15], 
                   max_seq_len=72, use_augmentation=True, seed=42):
    """
    创建训练、验证和测试数据集
    
    参数:
        data_path: 数据目录
        split_ratio: [训练,验证,测试]比例
        max_seq_len: 最大序列长度
        use_augmentation: 是否使用数据增强
        seed: 随机种子
        
    返回:
        train_dataset, val_dataset, test_dataset
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载处理后的数据
    aligned_data = pd.read_csv(os.path.join(data_path, 'processed/aligned_data.csv'))
    patient_info = pd.read_csv(os.path.join(data_path, 'processed/patient_info.csv'))
    
    # 合并脓毒症标签
    data = pd.merge(
        aligned_data,
        patient_info[['subject_id', 'sepsis_label', 'sepsis_onset_time']],
        on='subject_id',
        how='left'
    )
    
    # 按患者ID分组
    patient_groups = data.groupby('subject_id')
    
    # 提取特征
    vitals_columns = ['heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2']
    labs_columns = ['wbc', 'lactate', 'creatinine', 'platelet', 'bilirubin']
    drugs_columns = [col for col in data.columns if col.startswith('antibiotic_') or col.startswith('vasopressor_')]
    
    # 预处理数据
    vitals = []
    labs = []
    drugs = []
    text_embeds = []
    kg_embeds = []
    time_indices = []
    targets = []
    patient_ids = []
    
    # 用于文本嵌入的模拟数据
    # 在实际应用中，这应该来自处理好的文本嵌入
    text_embed_dim = 768
    kg_embed_dim = 64
    
    # 加载患者数据
    n_samples = 0
    max_samples = 500000  # 最大处理的样本数，防止OOM
    
    print(f"处理患者数据，最大样本数: {max_samples}")
    for patient_id, group in patient_groups:
        # 按时间排序
        group = group.sort_values('charttime')
        
        # 提取特征
        patient_vitals = group[vitals_columns].values
        patient_labs = group[labs_columns].values
        patient_drugs = group[drugs_columns].values if drugs_columns else np.zeros((len(group), 0))
        
        # 创建时间索引
        patient_time_indices = np.arange(1, len(group) + 1)
        
        # 创建目标值 - 如果脓毒症发作时间在当前时间之后，目标为1
        sepsis_label = group['sepsis_label'].iloc[0]
        sepsis_onset_time = group['sepsis_onset_time'].iloc[0] if not pd.isna(group['sepsis_onset_time'].iloc[0]) else None
        
        if sepsis_label == 1 and sepsis_onset_time is not None:
            # 将sepsis_onset_time转换为与charttime相同的格式
            onset_time = pd.to_datetime(sepsis_onset_time)
            charttimes = pd.to_datetime(group['charttime'])
            
            # 计算每个时间点是否已经发生脓毒症
            patient_targets = (charttimes >= onset_time).astype(float).values
        else:
            # 非脓毒症患者，目标全为0
            patient_targets = np.zeros(len(group))
        
        # 填充到最大序列长度
        if len(group) < max_seq_len:
            # 填充
            padded_vitals = np.zeros((max_seq_len, len(vitals_columns)))
            padded_labs = np.zeros((max_seq_len, len(labs_columns)))
            padded_drugs = np.zeros((max_seq_len, patient_drugs.shape[1]))
            padded_time_indices = np.zeros(max_seq_len)
            padded_targets = np.zeros(max_seq_len)
            
            # 复制有效数据
            padded_vitals[:len(group)] = patient_vitals
            padded_labs[:len(group)] = patient_labs
            padded_drugs[:len(group)] = patient_drugs
            padded_time_indices[:len(group)] = patient_time_indices
            padded_targets[:len(group)] = patient_targets
            
            patient_vitals = padded_vitals
            patient_labs = padded_labs
            patient_drugs = padded_drugs
            patient_time_indices = padded_time_indices
            patient_targets = padded_targets
        else:
            # 截断
            patient_vitals = patient_vitals[:max_seq_len]
            patient_labs = patient_labs[:max_seq_len]
            patient_drugs = patient_drugs[:max_seq_len]
            patient_time_indices = patient_time_indices[:max_seq_len]
            patient_targets = patient_targets[:max_seq_len]
        
        # 添加到列表
        vitals.append(patient_vitals)
        labs.append(patient_labs)
        drugs.append(patient_drugs)
        
        # 生成随机的文本和知识图谱嵌入
        # 在实际应用中，这应该是实际的嵌入
        text_embed = np.random.randn(text_embed_dim) * 0.1
        kg_embed = np.random.randn(kg_embed_dim) * 0.1
        
        text_embeds.append(text_embed)
        kg_embeds.append(kg_embed)
        
        time_indices.append(patient_time_indices)
        targets.append(patient_targets)
        patient_ids.append(str(patient_id))
        
        n_samples += 1
        if n_samples >= max_samples:
            break
    
    # 转换为numpy数组
    vitals = np.array(vitals)
    labs = np.array(labs)
    drugs = np.array(drugs)
    text_embeds = np.array(text_embeds)
    kg_embeds = np.array(kg_embeds)
    time_indices = np.array(time_indices)
    targets = np.array(targets)
    
    # 划分数据集
    n_samples = len(vitals)
    indices = np.random.permutation(n_samples)
    
    train_end = int(split_ratio[0] * n_samples)
    val_end = train_end + int(split_ratio[1] * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 创建数据集
    train_dataset = SepsisDataset(
        vitals=vitals[train_indices],
        labs=labs[train_indices],
        drugs=drugs[train_indices],
        text_embeds=text_embeds[train_indices],
        kg_embeds=kg_embeds[train_indices],
        time_indices=time_indices[train_indices],
        targets=targets[train_indices],
        patient_ids=[patient_ids[i] for i in train_indices],
        max_seq_len=max_seq_len,
        use_augmentation=use_augmentation
    )
    
    val_dataset = SepsisDataset(
        vitals=vitals[val_indices],
        labs=labs[val_indices],
        drugs=drugs[val_indices],
        text_embeds=text_embeds[val_indices],
        kg_embeds=kg_embeds[val_indices],
        time_indices=time_indices[val_indices],
        targets=targets[val_indices],
        patient_ids=[patient_ids[i] for i in val_indices],
        max_seq_len=max_seq_len,
        use_augmentation=False  # 验证集不使用增强
    )
    
    test_dataset = SepsisDataset(
        vitals=vitals[test_indices],
        labs=labs[test_indices],
        drugs=drugs[test_indices],
        text_embeds=text_embeds[test_indices],
        kg_embeds=kg_embeds[test_indices],
        time_indices=time_indices[test_indices],
        targets=targets[test_indices],
        patient_ids=[patient_ids[i] for i in test_indices],
        max_seq_len=max_seq_len,
        use_augmentation=False  # 测试集不使用增强
    )
    
    print(f"创建完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return train_dataset, val_dataset, test_dataset 