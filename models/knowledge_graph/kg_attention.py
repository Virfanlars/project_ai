import torch
import torch.nn as nn
import torch.nn.functional as F

class KGAttention(nn.Module):
    """
    知识图谱注意力融合模块
    用于将知识图谱信息融入患者特征
    支持处理时间序列数据
    """
    def __init__(self, feature_dim, kg_dim, hidden_dim=64):
        super(KGAttention, self).__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.kg_proj = nn.Linear(kg_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output_proj = nn.Linear(feature_dim + kg_dim, feature_dim)
        
    def forward(self, patient_features, kg_embeddings, kg_mask=None):
        """
        将知识图谱信息融入患者特征
        
        参数:
        - patient_features: 患者特征 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        - kg_embeddings: 知识图谱嵌入 [batch_size, num_entities, kg_dim] 或 [num_entities, kg_dim]
        - kg_mask: 可选的掩码，标识哪些实体应被关注 [num_entities]
        
        返回:
        - 融合后的特征 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        """
        # 检测输入维度
        is_sequence = len(patient_features.shape) == 3
        
        if is_sequence:
            # 处理序列数据
            batch_size, seq_len, feature_dim = patient_features.shape
            
            # 首先将序列特征展平为二维 [batch_size*seq_len, feature_dim]
            flat_features = patient_features.contiguous().view(-1, feature_dim)
            
            # 处理知识图谱嵌入
            if len(kg_embeddings.shape) == 3:  # [batch_size, num_entities, kg_dim]
                # 对于批量特定的KG嵌入，需要扩展为 [batch_size*seq_len, num_entities, kg_dim]
                num_entities, kg_dim = kg_embeddings.shape[1], kg_embeddings.shape[2]
                
                # 重复每个批次的KG嵌入seq_len次
                expanded_kg = kg_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
                expanded_kg = expanded_kg.contiguous().view(batch_size*seq_len, num_entities, kg_dim)
                
                # 使用扩展后的KG嵌入计算注意力
                enhanced_features, attn_weights = self._apply_attention(
                    flat_features, expanded_kg, kg_mask, batch_size*seq_len
                )
            else:  # [num_entities, kg_dim]
                num_entities, kg_dim = kg_embeddings.shape
                
                # 使用相同的KG嵌入为所有批次计算注意力
                enhanced_features, attn_weights = self._apply_attention(
                    flat_features, kg_embeddings, kg_mask, batch_size*seq_len
                )
            
            # 将结果重新整形为序列形式 [batch_size, seq_len, feature_dim]
            enhanced_features = enhanced_features.view(batch_size, seq_len, -1)
            attn_weights = attn_weights.view(batch_size, seq_len, -1)
        else:
            # 处理非序列数据（原始实现）
            batch_size = patient_features.shape[0]
            
            if len(kg_embeddings.shape) == 3:  # [batch_size, num_entities, kg_dim]
                enhanced_features, attn_weights = self._apply_attention(
                    patient_features, kg_embeddings.squeeze(1), kg_mask, batch_size
                )
            else:  # [num_entities, kg_dim]
                enhanced_features, attn_weights = self._apply_attention(
                    patient_features, kg_embeddings, kg_mask, batch_size
                )
                
        return enhanced_features, attn_weights
    
    def _apply_attention(self, features, kg_embeddings, kg_mask, batch_size):
        """
        应用注意力机制融合特征和知识图谱
        
        参数:
        - features: 特征 [batch_size, feature_dim]
        - kg_embeddings: 知识图谱嵌入 [num_entities, kg_dim] 或 [batch_size, num_entities, kg_dim]
        - kg_mask: 可选的掩码
        - batch_size: 批次大小
        
        返回:
        - 融合后的特征 [batch_size, feature_dim]
        - 注意力权重 [batch_size, num_entities]
        """
        # 确定知识图谱嵌入维度
        is_batched_kg = len(kg_embeddings.shape) == 3
        
        if is_batched_kg:
            num_entities, kg_dim = kg_embeddings.shape[1], kg_embeddings.shape[2]
        else:
            num_entities, kg_dim = kg_embeddings.shape
        
        # 投影到同一空间
        patient_hidden = self.feature_proj(features)  # [batch_size, hidden_dim]
        
        if is_batched_kg:
            kg_hidden = self.kg_proj(kg_embeddings)  # [batch_size, num_entities, hidden_dim]
            
            # 计算注意力分数 - 批量化版本
            patient_expanded = patient_hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 拼接特征和KG嵌入
            attn_input = torch.cat([
                patient_expanded.expand(-1, num_entities, -1), 
                kg_hidden
            ], dim=2)  # [batch_size, num_entities, hidden_dim*2]
        else:
            kg_hidden = self.kg_proj(kg_embeddings)  # [num_entities, hidden_dim]
            
            # 扩展维度以计算注意力
            patient_expanded = patient_hidden.unsqueeze(1).expand(-1, num_entities, -1)  # [batch_size, num_entities, hidden_dim]
            kg_expanded = kg_hidden.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_entities, hidden_dim]
            
            # 计算注意力分数
            attn_input = torch.cat([patient_expanded, kg_expanded], dim=2)  # [batch_size, num_entities, hidden_dim*2]
        
        attn_scores = self.attention(attn_input).squeeze(-1)  # [batch_size, num_entities]
        
        # 应用掩码(如果提供)
        if kg_mask is not None:
            kg_mask = kg_mask.unsqueeze(0).expand(batch_size, -1)
            attn_scores = attn_scores.masked_fill(~kg_mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, num_entities]
        
        # 加权聚合知识图谱信息
        if is_batched_kg:
            kg_context = torch.bmm(attn_weights.unsqueeze(1), kg_embeddings).squeeze(1)
        else:
            kg_embeddings_expanded = kg_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            kg_context = torch.bmm(attn_weights.unsqueeze(1), kg_embeddings_expanded).squeeze(1)
        
        # 融合患者特征和知识图谱上下文
        combined_features = torch.cat([features, kg_context], dim=1)
        output_features = self.output_proj(combined_features)
        
        return output_features, attn_weights