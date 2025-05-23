import torch
import torch.nn as nn
import torch.nn.functional as F

class SepsisTransformerModel(nn.Module):
    def __init__(self, vitals_dim, lab_dim, drug_dim, text_dim, kg_dim, 
                 hidden_dim=64, num_heads=2, num_layers=1, dropout=0.2):
        super(SepsisTransformerModel, self).__init__()
        
        # 各模态特征维度
        self.vitals_dim = vitals_dim
        self.lab_dim = lab_dim
        self.drug_dim = drug_dim
        self.text_dim = text_dim
        self.kg_dim = kg_dim
        
        # 模态特征投影层
        self.vitals_proj = nn.Sequential(
            nn.Linear(vitals_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lab_proj = nn.Sequential(
            nn.Linear(lab_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.position_encoder = nn.Embedding(24*7, hidden_dim)  # 假设最长为一周的数据
        
        # 简化: 使用较少的Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,  # 减小前馈网络大小
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()  # 保留sigmoid获得概率输出
        )
        
        # 简化知识图谱处理 - 使用简单投影
        self.kg_proj = nn.Linear(kg_dim, hidden_dim)
    
    def forward(self, vitals, labs, drugs, text_embed, kg_embed, time_indices):
        batch_size, seq_len, _ = vitals.shape
        
        # 重塑张量以适应BatchNorm1d (expects [batch, features])
        # 处理各模态特征投影
        vitals_flat = vitals.reshape(-1, self.vitals_dim)  # [batch*seq, vitals_dim]
        vitals_proj_flat = self.vitals_proj(vitals_flat)  # [batch*seq, hidden]
        vitals_proj = vitals_proj_flat.reshape(batch_size, seq_len, -1)  # [batch, seq, hidden]
        
        labs_flat = labs.reshape(-1, self.lab_dim)
        labs_proj_flat = self.lab_proj(labs_flat)
        labs_proj = labs_proj_flat.reshape(batch_size, seq_len, -1)
        
        drugs_flat = drugs.reshape(-1, self.drug_dim)
        drugs_proj_flat = self.drug_proj(drugs_flat)
        drugs_proj = drugs_proj_flat.reshape(batch_size, seq_len, -1)
        
        # 处理文本嵌入 (适应改变)
        if text_embed.dim() == 3:  # [batch, seq, text_dim]
            text_flat = text_embed.reshape(-1, self.text_dim)
            text_proj_flat = self.text_proj(text_flat)
            text_proj = text_proj_flat.reshape(batch_size, seq_len, -1)
        else:  # [batch, text_dim]
            text_proj_flat = self.text_proj(text_embed)  # [batch, hidden]
            text_proj = text_proj_flat.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, hidden]
        
        # 简化知识图谱处理
        kg_proj_flat = self.kg_proj(kg_embed)  # [batch, hidden]
        kg_proj = kg_proj_flat.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, hidden]
        
        # 特征融合
        fused_features = vitals_proj + labs_proj + drugs_proj + text_proj + kg_proj
        
        # 添加位置编码
        position_encoded = self.position_encoder(time_indices)
        features = fused_features + position_encoded
        
        # 创建注意力掩码，忽略填充的时间点
        padding_mask = (time_indices == 0)
        
        # Transformer处理时序信息
        transformer_out = self.transformer(features, src_key_padding_mask=padding_mask)
        
        # 输出每个时间点的预测概率
        # 重塑以适应BatchNorm1d
        out_flat = transformer_out.reshape(-1, transformer_out.size(-1))
        out_flat = self.output_layer(out_flat)
        outputs = out_flat.reshape(batch_size, seq_len, -1)
        
        return outputs