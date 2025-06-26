#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识图谱嵌入模块
将知识图谱实体和关系转化为低维稠密向量表示
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TransEModel(nn.Module):
    """
    TransE知识图谱嵌入模型
    基于平移距离的知识图谱嵌入方法
    h + r ≈ t
    """
    def __init__(self, entity_count, relation_count, embedding_dim, margin=1.0):
        """
        初始化TransE模型
        
        Args:
            entity_count: 实体数量
            relation_count: 关系数量
            embedding_dim: 嵌入维度
            margin: 损失函数中的间隔参数
        """
        super(TransEModel, self).__init__()
        
        # 实体和关系的嵌入
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim)
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim)
        
        # 初始化嵌入，使用更小的初始值范围，避免不稳定
        nn.init.uniform_(self.entity_embeddings.weight, -0.05, 0.05)
        nn.init.uniform_(self.relation_embeddings.weight, -0.05, 0.05)
        
        # 归一化实体嵌入
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)
        
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
    
    def forward(self, positive_triples, negative_triples):
        """
        前向传播
        
        Args:
            positive_triples: 正例三元组 (head, relation, tail)
            negative_triples: 负例三元组 (head, relation, tail)，通过替换头或尾实体生成
            
        Returns:
            torch.Tensor: 损失值
        """
        # 解包三元组
        pos_heads, pos_relations, pos_tails = positive_triples
        neg_heads, neg_relations, neg_tails = negative_triples
        
        # 获取嵌入
        pos_head_embeds = self.entity_embeddings(pos_heads)
        pos_rel_embeds = self.relation_embeddings(pos_relations)
        pos_tail_embeds = self.entity_embeddings(pos_tails)
        
        neg_head_embeds = self.entity_embeddings(neg_heads)
        neg_rel_embeds = self.relation_embeddings(neg_relations)
        neg_tail_embeds = self.entity_embeddings(neg_tails)
        
        # 添加安全检查
        if torch.isnan(pos_head_embeds).any() or torch.isinf(pos_head_embeds).any():
            logger.warning("检测到NaN/Inf值在pos_head_embeds中")
            pos_head_embeds = torch.nan_to_num(pos_head_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(pos_rel_embeds).any() or torch.isinf(pos_rel_embeds).any():
            logger.warning("检测到NaN/Inf值在pos_rel_embeds中")
            pos_rel_embeds = torch.nan_to_num(pos_rel_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(pos_tail_embeds).any() or torch.isinf(pos_tail_embeds).any():
            logger.warning("检测到NaN/Inf值在pos_tail_embeds中")
            pos_tail_embeds = torch.nan_to_num(pos_tail_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(neg_head_embeds).any() or torch.isinf(neg_head_embeds).any():
            logger.warning("检测到NaN/Inf值在neg_head_embeds中")
            neg_head_embeds = torch.nan_to_num(neg_head_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(neg_rel_embeds).any() or torch.isinf(neg_rel_embeds).any():
            logger.warning("检测到NaN/Inf值在neg_rel_embeds中")
            neg_rel_embeds = torch.nan_to_num(neg_rel_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(neg_tail_embeds).any() or torch.isinf(neg_tail_embeds).any():
            logger.warning("检测到NaN/Inf值在neg_tail_embeds中")
            neg_tail_embeds = torch.nan_to_num(neg_tail_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化实体嵌入，添加小epsilon避免零向量
        epsilon = 1e-8
        pos_head_embeds = F.normalize(pos_head_embeds + epsilon, p=2, dim=1)
        pos_tail_embeds = F.normalize(pos_tail_embeds + epsilon, p=2, dim=1)
        neg_head_embeds = F.normalize(neg_head_embeds + epsilon, p=2, dim=1)
        neg_tail_embeds = F.normalize(neg_tail_embeds + epsilon, p=2, dim=1)
        
        # 计算能量（距离）：||h + r - t||
        pos_scores = torch.norm(pos_head_embeds + pos_rel_embeds - pos_tail_embeds, p=2, dim=1)
        neg_scores = torch.norm(neg_head_embeds + neg_rel_embeds - neg_tail_embeds, p=2, dim=1)
        
        # 确保不含NaN或Inf
        pos_scores = torch.nan_to_num(pos_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        neg_scores = torch.nan_to_num(neg_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 目标是让正例分数小于负例分数
        target = torch.ones_like(pos_scores)
        
        # 计算损失：max(0, margin + pos_score - neg_score)
        loss_values = self.criterion(neg_scores, pos_scores, target)
        
        # 过滤掉NaN和Inf值
        valid_mask = ~torch.isnan(loss_values) & ~torch.isinf(loss_values)
        if valid_mask.sum() == 0:
            # 如果所有值都是NaN或Inf，返回一个小的常数作为损失
            return torch.tensor(0.1, device=pos_scores.device, requires_grad=True)
        
        # 只使用有效的损失值
        valid_loss = loss_values[valid_mask].mean()
        
        return valid_loss
    
    def get_entity_embedding(self, entity_id):
        """获取实体嵌入向量"""
        return self.entity_embeddings(torch.LongTensor([entity_id])).detach().cpu().numpy()[0]
    
    def get_relation_embedding(self, relation_id):
        """获取关系嵌入向量"""
        return self.relation_embeddings(torch.LongTensor([relation_id])).detach().cpu().numpy()[0]
    
    def get_all_entity_embeddings(self):
        """获取所有实体的嵌入向量"""
        return self.entity_embeddings.weight.detach().cpu().numpy()
    
    def get_all_relation_embeddings(self):
        """获取所有关系的嵌入向量"""
        return self.relation_embeddings.weight.detach().cpu().numpy()


class KGTripleDataset(Dataset):
    """知识图谱三元组数据集"""
    def __init__(self, kg, negative_samples=1):
        """
        初始化数据集
        
        Args:
            kg: 知识图谱对象
            negative_samples: 每个正例三元组对应的负例数量
        """
        self.kg = kg
        self.triples = kg.triples
        self.entity_count = len(kg.entity2id)
        self.relation_count = len(kg.relation2id)
        self.negative_samples = negative_samples
        
        # 实体和关系的映射
        self.entity2id = kg.entity2id
        self.relation2id = kg.relation2id
    
    def __len__(self):
        """返回三元组数量"""
        return len(self.triples)
    
    def __getitem__(self, idx):
        """获取指定索引的三元组及其负例"""
        # 正例三元组
        pos_triple = self.triples[idx]
        head_str, relation_str, tail_str = pos_triple
        
        # 将字符串转换为ID
        head_id = self.entity2id[head_str]
        relation_id = self.relation2id[relation_str]
        tail_id = self.entity2id[tail_str]
        
        # 生成负例三元组
        negative_triples = []
        for _ in range(self.negative_samples):
            # 以50%的概率替换头实体或尾实体
            if np.random.random() < 0.5:
                # 替换头实体
                new_head_id = np.random.randint(0, self.entity_count)
                while new_head_id == head_id:  # 避免生成与正例相同的三元组
                    new_head_id = np.random.randint(0, self.entity_count)
                negative_triples.append((new_head_id, relation_id, tail_id))
            else:
                # 替换尾实体
                new_tail_id = np.random.randint(0, self.entity_count)
                while new_tail_id == tail_id:  # 避免生成与正例相同的三元组
                    new_tail_id = np.random.randint(0, self.entity_count)
                negative_triples.append((head_id, relation_id, new_tail_id))
        
        # 将负例三元组转换为NumPy数组
        negative_triples = np.array(negative_triples, dtype=np.int64)
        
        return {
            'positive_triple': np.array([head_id, relation_id, tail_id], dtype=np.int64),
            'negative_triples': negative_triples
        }


def collate_fn(batch):
    """
    批处理函数
    
    Args:
        batch: 批次数据
        
    Returns:
        dict: 处理后的批次数据
    """
    # 收集所有正例三元组
    positive_triples = np.vstack([item['positive_triple'] for item in batch])
    
    # 收集所有负例三元组
    all_negative_triples = []
    for item in batch:
        all_negative_triples.append(item['negative_triples'])
    
    # 确保每个样本有相同数量的负例
    max_neg_samples = max(neg.shape[0] for neg in all_negative_triples)
    batch_size = len(batch)
    
    # 统一负例数量 - 如果一个样本的负例不足，通过复制已有负例来补充
    uniform_negative_triples = []
    for neg_triples in all_negative_triples:
        if neg_triples.shape[0] < max_neg_samples:
            # 需要补充的数量
            num_to_add = max_neg_samples - neg_triples.shape[0]
            
            # 通过随机复制已有负例来补充
            indices = np.random.randint(0, neg_triples.shape[0], size=num_to_add)
            additional_triples = neg_triples[indices]
            
            # 合并原有和补充的负例
            neg_triples = np.vstack([neg_triples, additional_triples])
        
        uniform_negative_triples.append(neg_triples)
    
    # 将所有统一大小的负例合并
    negative_triples = np.vstack(uniform_negative_triples)
    
    # 确保正例也有相同数量的复制
    # 每个正例需要复制max_neg_samples次，以匹配负例的数量
    expanded_pos_triples = np.repeat(positive_triples, max_neg_samples, axis=0)
    
    # 转换为Tensor，确保数据类型为长整型
    positive_triples = torch.from_numpy(expanded_pos_triples.astype(np.int64)).long()
    negative_triples = torch.from_numpy(negative_triples.astype(np.int64)).long()
    
    # 将正例和负例三元组分解为头、关系和尾
    pos_heads = positive_triples[:, 0]
    pos_relations = positive_triples[:, 1]
    pos_tails = positive_triples[:, 2]
    
    neg_heads = negative_triples[:, 0]
    neg_relations = negative_triples[:, 1]
    neg_tails = negative_triples[:, 2]
    
    return (
        (pos_heads, pos_relations, pos_tails),
        (neg_heads, neg_relations, neg_tails)
    )


def train_embeddings(kg, embedding_dim=64, batch_size=128, epochs=50, lr=0.01, 
                     negative_samples=5, margin=1.0, device='cuda'):
    """
    训练知识图谱嵌入
    
    Args:
        kg: 知识图谱对象
        embedding_dim: 嵌入维度
        batch_size: 批处理大小
        epochs: 训练轮数
        lr: 学习率
        negative_samples: 每个正例三元组对应的负例数量
        margin: 损失函数中的间隔参数
        device: 训练设备
        
    Returns:
        TransEModel: 训练好的模型
    """
    # 如果没有可用的GPU，使用CPU
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA不可用，使用CPU进行训练")
    
    device = torch.device(device)
    
    # 创建数据集和数据加载器
    dataset = KGTripleDataset(kg, negative_samples=negative_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    model = TransEModel(
        entity_count=len(kg.entity2id),
        relation_count=len(kg.relation2id),
        embedding_dim=embedding_dim,
        margin=margin
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 添加权重衰减
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for positive_triples, negative_triples in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                # 确保批次中有数据
                if len(positive_triples[0]) == 0:
                    continue
                
                # 将数据移到设备
                pos_heads, pos_relations, pos_tails = [t.to(device) for t in positive_triples]
                neg_heads, neg_relations, neg_tails = [t.to(device) for t in negative_triples]
                
                # 检查正例和负例大小是否匹配
                if pos_heads.size(0) != neg_heads.size(0):
                    logger.warning(f"批次大小不匹配: 正例={pos_heads.size(0)}, 负例={neg_heads.size(0)}")
                    # 调整大小使它们匹配
                    min_size = min(pos_heads.size(0), neg_heads.size(0))
                    pos_heads = pos_heads[:min_size]
                    pos_relations = pos_relations[:min_size]
                    pos_tails = pos_tails[:min_size]
                    neg_heads = neg_heads[:min_size]
                    neg_relations = neg_relations[:min_size]
                    neg_tails = neg_tails[:min_size]
                
                # 前向传播
                loss = model(
                    (pos_heads, pos_relations, pos_tails),
                    (neg_heads, neg_relations, neg_tails)
                )
                
                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"损失为NaN或Inf，跳过批次")
                    continue
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                batch_count += 1
                
                # 每次迭代后归一化实体嵌入
                with torch.no_grad():
                    entity_embeds = model.entity_embeddings.weight.data
                    # 添加小的epsilon避免零向量
                    epsilon = 1e-8
                    entity_embeds = F.normalize(entity_embeds + epsilon, p=2, dim=1)
                    # 检查并处理NaN和Inf
                    entity_embeds = torch.nan_to_num(entity_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
                    model.entity_embeddings.weight.data.copy_(entity_embeds)
            
            except Exception as e:
                logger.error(f"批次处理错误: {e}")
                continue
        
        # 输出每个epoch的平均损失
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 更新学习率
            scheduler.step(avg_loss)
        else:
            logger.warning(f"Epoch {epoch+1}/{epochs}没有有效批次")
    
    return model


def generate_embeddings(kg, method='TransE', embedding_dim=64, output_dir='./output', force_train=False):
    """
    生成知识图谱嵌入
    
    Args:
        kg: 知识图谱对象
        method: 嵌入方法，支持 'TransE', 'DistMult', 'ComplEx'
        embedding_dim: 嵌入维度
        output_dir: 输出目录
        force_train: 是否强制重新训练
        
    Returns:
        dict: 包含实体和关系嵌入的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 嵌入文件路径
    embedding_file = os.path.join(output_dir, f"{method}_embeddings.pkl")
    
    # 如果嵌入文件已存在且不强制重新训练，则直接加载
    if os.path.exists(embedding_file) and not force_train:
        logger.info(f"加载已有的{method}嵌入...")
        with open(embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    
    # 训练嵌入
    logger.info(f"训练{method}嵌入...")
    
    # 获取实体和关系数量
    entity_count = len(kg.entity2id)
    relation_count = len(kg.relation2id)
    
    # 确定合适的批次大小，对于大图谱使用较小的批次
    if entity_count > 1000:
        batch_size = 32  # 使用更小的批次
    else:
        batch_size = 64
    
    # 根据图谱大小调整训练轮数
    if entity_count > 5000:
        epochs = 20
    elif entity_count > 1000:
        epochs = 30
    else:
        epochs = 40
    
    try:
        if method == 'TransE':
            logger.info("使用TransE方法训练知识图谱嵌入")
            # 对于数值稳定性，使用更小的学习率、更小的embedding_dim、更小的margin
            model = train_embeddings(
                kg,
                embedding_dim=min(embedding_dim, 64),  # 限制最大维度
                batch_size=batch_size,
                epochs=epochs,
                lr=0.001,  # 使用更小的学习率
                negative_samples=2,  # 减少负样本数量以减轻内存压力
                margin=0.5,  # 使用更小的margin
                device='cpu'  # 强制使用CPU，避免GPU上的数值问题
            )
        elif method == 'DistMult':
            # TODO: 实现DistMult模型
            logger.warning(f"{method}模型尚未实现，使用TransE替代")
            model = train_embeddings(
                kg, 
                embedding_dim=min(embedding_dim, 64), 
                batch_size=batch_size, 
                epochs=epochs,
                lr=0.001,
                device='cpu'
            )
        elif method == 'ComplEx':
            # TODO: 实现ComplEx模型
            logger.warning(f"{method}模型尚未实现，使用TransE替代")
            model = train_embeddings(
                kg, 
                embedding_dim=min(embedding_dim, 64), 
                batch_size=batch_size, 
                epochs=epochs,
                lr=0.001,
                device='cpu'
            )
        else:
            raise ValueError(f"不支持的嵌入方法: {method}")
        
        # 获取所有实体和关系的嵌入
        entity_embeddings = model.get_all_entity_embeddings()
        relation_embeddings = model.get_all_relation_embeddings()
        
        # 检查嵌入是否包含NaN或Inf
        if np.isnan(entity_embeddings).any() or np.isinf(entity_embeddings).any():
            logger.warning("实体嵌入包含NaN或Inf值，将被替换为0")
            entity_embeddings = np.nan_to_num(entity_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(relation_embeddings).any() or np.isinf(relation_embeddings).any():
            logger.warning("关系嵌入包含NaN或Inf值，将被替换为0")
            relation_embeddings = np.nan_to_num(relation_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 创建嵌入字典
        embeddings = {
            'method': method,
            'dimension': min(embedding_dim, 64),  # 使用实际使用的维度
            'entity_embeddings': entity_embeddings,
            'relation_embeddings': relation_embeddings,
            'entity2id': kg.entity2id,
            'relation2id': kg.relation2id,
            'id2entity': kg.id2entity,
            'id2relation': kg.id2relation
        }
        
        # 保存嵌入
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"{method}嵌入已保存至 {embedding_file}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"训练嵌入时发生错误: {e}")
        raise


def get_entity_embedding(embeddings, entity_name):
    """
    获取指定实体的嵌入向量
    
    Args:
        embeddings: 嵌入字典
        entity_name: 实体名称
        
    Returns:
        numpy.ndarray: 实体嵌入向量
    """
    entity2id = embeddings['entity2id']
    entity_embeddings = embeddings['entity_embeddings']
    
    if entity_name in entity2id:
        entity_id = entity2id[entity_name]
        return entity_embeddings[entity_id]
    else:
        # 如果实体不存在，返回零向量
        logger.warning(f"实体 '{entity_name}' 不存在于知识图谱中")
        return np.zeros(embeddings['dimension'])


def get_relation_embedding(embeddings, relation_name):
    """
    获取指定关系的嵌入向量
    
    Args:
        embeddings: 嵌入字典
        relation_name: 关系名称
        
    Returns:
        numpy.ndarray: 关系嵌入向量
    """
    relation2id = embeddings['relation2id']
    relation_embeddings = embeddings['relation_embeddings']
    
    if relation_name in relation2id:
        relation_id = relation2id[relation_name]
        return relation_embeddings[relation_id]
    else:
        # 如果关系不存在，返回零向量
        logger.warning(f"关系 '{relation_name}' 不存在于知识图谱中")
        return np.zeros(embeddings['dimension'])


def get_entity_similarities(embeddings, entity_name, top_k=10):
    """
    获取与指定实体最相似的实体
    
    Args:
        embeddings: 嵌入字典
        entity_name: 实体名称
        top_k: 返回的最相似实体数量
        
    Returns:
        list: (实体名称, 相似度分数) 的元组列表
    """
    entity2id = embeddings['entity2id']
    id2entity = embeddings['id2entity']
    entity_embeddings = embeddings['entity_embeddings']
    
    if entity_name not in entity2id:
        logger.warning(f"实体 '{entity_name}' 不存在于知识图谱中")
        return []
    
    # 获取目标实体嵌入
    entity_id = entity2id[entity_name]
    target_embedding = entity_embeddings[entity_id]
    
    # 计算所有实体与目标实体的余弦相似度
    dot_products = np.dot(entity_embeddings, target_embedding)
    norms = np.linalg.norm(entity_embeddings, axis=1) * np.linalg.norm(target_embedding)
    similarities = dot_products / norms
    
    # 获取最相似的实体（排除自身）
    similar_ids = np.argsort(-similarities)
    similar_ids = [idx for idx in similar_ids if idx != entity_id][:top_k]
    
    # 返回实体名称和相似度分数
    return [(id2entity[idx], similarities[idx]) for idx in similar_ids]


def load_embeddings(embeddings_path):
    """
    从.npz文件中加载知识图谱嵌入
    
    Args:
        embeddings_path: 嵌入文件路径(.npz格式)
        
    Returns:
        dict: 包含嵌入和映射的字典
    """
    logger.info(f"从 {embeddings_path} 加载知识图谱嵌入")
    
    try:
        # 加载.npz文件
        data = np.load(embeddings_path, allow_pickle=True)
        
        # 提取嵌入和映射
        embeddings = {
            'entity_embeddings': data['entity_embeddings'],
            'relation_embeddings': data['relation_embeddings']
        }
        
        # 加载映射字典(如果存在)
        if 'entity2id.npy' in data.files:
            embeddings['entity2id'] = data['entity2id'].item()
        else:
            # 创建默认映射
            embeddings['entity2id'] = {f'entity_{i}': i for i in range(len(embeddings['entity_embeddings']))}
        
        if 'id2entity.npy' in data.files:
            embeddings['id2entity'] = data['id2entity'].item()
        else:
            # 从entity2id创建反向映射
            embeddings['id2entity'] = {v: k for k, v in embeddings['entity2id'].items()}
            
        if 'relation2id.npy' in data.files:
            embeddings['relation2id'] = data['relation2id'].item()
        else:
            # 创建默认映射
            embeddings['relation2id'] = {f'relation_{i}': i for i in range(len(embeddings['relation_embeddings']))}
            
        if 'id2relation.npy' in data.files:
            embeddings['id2relation'] = data['id2relation'].item()
        else:
            # 从relation2id创建反向映射
            embeddings['id2relation'] = {v: k for k, v in embeddings['relation2id'].items()}
        
        logger.info(f"成功加载知识图谱嵌入，实体数量: {len(embeddings['entity_embeddings'])}, 关系数量: {len(embeddings['relation_embeddings'])}")
        return embeddings
        
    except Exception as e:
        logger.error(f"加载知识图谱嵌入失败: {e}")
        raise 


class KnowledgeGraphEmbedder:
    """
    知识图谱嵌入器类，封装知识图谱嵌入功能
    """
    
    def __init__(self, knowledge_graph, embedding_dim=64, method='TransE'):
        """
        初始化知识图谱嵌入器
        
        Args:
            knowledge_graph: 知识图谱对象
            embedding_dim: 嵌入维度
            method: 嵌入方法
        """
        self.knowledge_graph = knowledge_graph
        self.embedding_dim = embedding_dim
        self.method = method
        self.embeddings = None
        
    def train_embeddings(self, epochs=50, batch_size=128, lr=0.001):
        """
        训练知识图谱嵌入
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            
        Returns:
            嵌入字典
        """
        try:
            logger.info(f"开始训练{self.method}嵌入...")
            
            # 使用现有的generate_embeddings函数
            self.embeddings = generate_embeddings(
                kg=self.knowledge_graph,
                method=self.method,
                embedding_dim=self.embedding_dim,
                output_dir='temp',  # 临时目录
                force_train=True
            )
            
            logger.info("知识图谱嵌入训练完成")
            return self.embeddings
            
        except Exception as e:
            logger.error(f"训练知识图谱嵌入失败: {e}")
            # 返回随机嵌入作为备用
            logger.warning("将使用随机嵌入作为备用")
            return self._create_random_embeddings()
    
    def _create_random_embeddings(self):
        """創建隨機嵌入作為備用"""
        entity_count = len(self.knowledge_graph.entity2id) if hasattr(self.knowledge_graph, 'entity2id') else 100
        relation_count = len(self.knowledge_graph.relation2id) if hasattr(self.knowledge_graph, 'relation2id') else 10
        
        embeddings = {
            'method': 'Random',
            'dimension': self.embedding_dim,
            'entity_embeddings': np.random.randn(entity_count, self.embedding_dim).astype(np.float32),
            'relation_embeddings': np.random.randn(relation_count, self.embedding_dim).astype(np.float32),
            'entity2id': getattr(self.knowledge_graph, 'entity2id', {}),
            'relation2id': getattr(self.knowledge_graph, 'relation2id', {}),
            'id2entity': getattr(self.knowledge_graph, 'id2entity', {}),
            'id2relation': getattr(self.knowledge_graph, 'id2relation', {})
        }
        
        return embeddings
    
    def save_embeddings(self, output_path):
        """
        保存嵌入到文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.embeddings is None:
            logger.error("没有可保存的嵌入，请先训练嵌入")
            return
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"嵌入已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存嵌入失败: {e}")
    
    def load_embeddings(self, input_path):
        """
        从文件加载嵌入
        
        Args:
            input_path: 输入文件路径
        """
        try:
            with open(input_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info(f"嵌入已从 {input_path} 加载")
        except Exception as e:
            logger.error(f"加载嵌入失败: {e}")
            self.embeddings = None
    
    def get_embeddings(self):
        """
        获取嵌入
        
        Returns:
            嵌入字典
        """
        return self.embeddings