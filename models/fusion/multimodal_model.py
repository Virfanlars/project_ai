import torch
import torch.nn as nn
import torch.nn.functional as F
from models.knowledge_graph.kg_attention import KGAttention

class SepsisTransformerModel(nn.Module):
    def __init__(self, vitals_dim, lab_dim, drug_dim, text_dim, kg_dim, 
                 hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(SepsisTransformerModel, self).__init__()
        
        # 各模态特征维度
        self.vitals_dim = vitals_dim
        self.lab_dim = lab_dim
        self.drug_dim = drug_dim
        self.text_dim = text_dim
        self.kg_dim = kg_dim
        
        # 模态特征投影层
        self.vitals_proj = nn.Linear(vitals_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 位置编码
        self.position_encoder = nn.Embedding(24*7, hidden_dim)  # 假设最长为一周的数据
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 替换简单的KG投影为注意力融合模块
        self.use_kg_attention = True  # 配置选项
        if self.use_kg_attention:
            self.kg_attention = KGAttention(
                feature_dim=hidden_dim,
                kg_dim=kg_dim,
                hidden_dim=hidden_dim//2
            )
        else:
            # 保留原有简单投影作为备选
            self.kg_proj = nn.Linear(kg_dim, hidden_dim)
    
    def forward(self, vitals, labs, drugs, text_embed, kg_embed, time_indices):
        # 各模态特征投影
        vitals_proj = self.vitals_proj(vitals)
        labs_proj = self.lab_proj(labs)
        drugs_proj = self.drug_proj(drugs)
        text_proj = self.text_proj(text_embed)
        
        # 特殊处理知识图谱嵌入 - 使用注意力机制
        batch_size, seq_len, _ = vitals_proj.shape
        
        if self.use_kg_attention:
            # 假设kg_embed是[batch_size, num_entities, kg_dim]格式
            # 如果不是，需要先进行格式转换
            if len(kg_embed.shape) == 2:
                # 如果是[batch_size, kg_dim]格式，扩展为单实体
                kg_embed = kg_embed.unsqueeze(1)  # [batch_size, 1, kg_dim]
            
            # 先融合其他临床特征
            clinical_features = vitals_proj + labs_proj + drugs_proj + text_proj
            
            # 使用KG注意力模块融合
            enhanced_features, kg_attn_weights = self.kg_attention(
                clinical_features, kg_embed
            )
            
            # 存储注意力权重以便解释(如需要)
            self._kg_attention_weights = kg_attn_weights
            
            # 使用增强特征进行预测
            fused_features = enhanced_features
        else:
            # 原有的简单融合方式
            kg_proj_flat = self.kg_proj(kg_embed)  # [batch_size, hidden_dim]
            kg_proj = kg_proj_flat.unsqueeze(1).expand(-1, seq_len, -1)
            fused_features = vitals_proj + labs_proj + drugs_proj + text_proj + kg_proj
        
        # 添加位置编码
        position_encoded = self.position_encoder(time_indices)
        features = fused_features + position_encoded
        
        # 创建注意力掩码，忽略填充的时间点
        padding_mask = (time_indices == 0)
        
        # Transformer处理时序信息
        transformer_out = self.transformer(features, src_key_padding_mask=padding_mask)
        
        # 输出每个时间点的预测概率
        outputs = self.output_layer(transformer_out)
        
        return outputs