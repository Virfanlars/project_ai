import torch
import torch.nn as nn
import torch.nn.functional as F

class KGAttention(nn.Module):
    """
    知识图谱注意力融合模块
    用于将知识图谱信息融入患者特征
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
        - patient_features: 患者特征 [batch_size, feature_dim]
        - kg_embeddings: 知识图谱嵌入 [num_entities, kg_dim]
        - kg_mask: 可选的掩码，标识哪些实体应被关注 [num_entities]
        
        返回:
        - 融合后的特征 [batch_size, feature_dim]
        """
        batch_size = patient_features.size(0)
        num_entities = kg_embeddings.size(0)
        
        # 投影到同一空间
        patient_hidden = self.feature_proj(patient_features)  # [batch_size, hidden_dim]
        kg_hidden = self.kg_proj(kg_embeddings)  # [num_entities, hidden_dim]
        
        # 扩展维度以计算注意力
        patient_expanded = patient_hidden.unsqueeze(1).expand(-1, num_entities, -1)  # [batch_size, num_entities, hidden_dim]
        kg_expanded = kg_hidden.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_entities, hidden_dim]
        
        # 计算注意力分数
        attn_input = torch.cat([patient_expanded, kg_expanded], dim=2)  # [batch_size, num_entities, hidden_dim*2]
        attn_scores = self.attention(attn_input).squeeze(-1)  # [batch_size, num_entities]
        
        # 应用掩码(如果提供)
        if kg_mask is not None:
            attn_scores = attn_scores.masked_fill(~kg_mask.unsqueeze(0), -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, num_entities]
        
        # 加权聚合知识图谱信息
        kg_context = torch.bmm(attn_weights.unsqueeze(1), kg_embeddings.unsqueeze(0).expand(batch_size, -1, -1)).squeeze(1)
        
        # 融合患者特征和知识图谱上下文
        combined_features = torch.cat([patient_features, kg_context], dim=1)
        output_features = self.output_proj(combined_features)
        
        return output_features, attn_weights