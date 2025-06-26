#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态Transformer模型
用于融合生命体征、实验室值、药物使用和知识图谱嵌入
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            dropout: Dropout比例
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区而非参数
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        # 检查并处理NaN和Inf值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 应用位置编码
        x = x + self.pe[:, :x.size(1), :]
        
        # 再次检查并处理NaN和Inf值
        output = self.dropout(x)
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return output


class SepsisTransformerModel(nn.Module):
    """
    脓毒症预测的多模态Transformer模型
    融合生命体征、实验室值、药物使用和知识图谱嵌入
    """
    def __init__(self, vitals_dim, lab_dim, drug_dim, text_dim, kg_dim, 
                 hidden_dim=64, num_heads=2, num_layers=1, dropout=0.2):
        """
        初始化模型
        
        Args:
            vitals_dim: 生命体征特征维度
            lab_dim: 实验室值特征维度
            drug_dim: 药物使用特征维度
            text_dim: 文本特征维度
            kg_dim: 知识图谱嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量
            num_layers: Transformer编码器层数
            dropout: Dropout比例
        """
        super(SepsisTransformerModel, self).__init__()
        
        # 特征投影层
        self.vitals_proj = nn.Linear(vitals_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.kg_proj = nn.Linear(kg_dim, hidden_dim)
        
        # 特征归一化层
        self.vitals_norm = nn.LayerNorm(hidden_dim)
        self.lab_norm = nn.LayerNorm(hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.kg_norm = nn.LayerNorm(hidden_dim)
        
        # 融合后的归一化层
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 时间嵌入层
        self.time_embedding = nn.Embedding(5000, hidden_dim)  # 最多支持5000小时
        
        # Transformer编码器 - 减少复杂度以提高稳定性
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,  # 减小前馈网络大小
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # 减少中间层维度
            nn.LeakyReLU(0.1),  # 使用LeakyReLU代替ReLU提高稳定性
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1之间的风险分数
        )
        
        # 数值稳定性的epsilon值
        self.eps = 1e-8
        
    def forward(self, vitals, labs, drugs, text_embed, kg_embed, time_indices, attention_mask=None):
        """
        前向传播
        
        Args:
            vitals: 生命体征特征 [batch_size, seq_len, vitals_dim]
            labs: 实验室值特征 [batch_size, seq_len, lab_dim]
            drugs: 药物使用特征 [batch_size, seq_len, drug_dim]
            text_embed: 文本嵌入 [batch_size, text_dim]
            kg_embed: 知识图谱嵌入 [batch_size, kg_dim]
            time_indices: 时间索引 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            torch.Tensor: 脓毒症风险分数 [batch_size, seq_len]
        """
        # 获取批次大小和序列长度
        batch_size, seq_len = vitals.size(0), vitals.size(1)
        
        # 确保序列长度不为零
        if seq_len == 0:
            logger.warning("输入序列长度为零，无法进行预测")
            return torch.zeros(batch_size, 1, device=vitals.device)
        
        # 检查并处理输入特征中的NaN和Inf值
        vitals = torch.nan_to_num(vitals, nan=0.0, posinf=1.0, neginf=-1.0)
        labs = torch.nan_to_num(labs, nan=0.0, posinf=1.0, neginf=-1.0)
        drugs = torch.nan_to_num(drugs, nan=0.0, posinf=1.0, neginf=-1.0)
        text_embed = torch.nan_to_num(text_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        kg_embed = torch.nan_to_num(kg_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 打印输入维度以进行调试
        logger.debug(f"输入维度检查 - vitals: {vitals.shape}, labs: {labs.shape}, drugs: {drugs.shape}")
        logger.debug(f"输入维度检查 - text_embed: {text_embed.shape}, kg_embed: {kg_embed.shape}")
        
        # 投影并归一化各模态特征
        try:
            # 检查并确保输入维度与投影层匹配
            if vitals.size(2) != self.vitals_proj.in_features:
                logger.warning(f"生命体征特征维度不匹配: 输入 {vitals.size(2)}, 预期 {self.vitals_proj.in_features}")
                # 创建适当维度的零张量并复制可用的特征
                valid_dim = min(vitals.size(2), self.vitals_proj.in_features)
                fixed_vitals = torch.zeros(batch_size, seq_len, self.vitals_proj.in_features, device=vitals.device)
                fixed_vitals[:, :, :valid_dim] = vitals[:, :, :valid_dim]
                vitals = fixed_vitals
                
            if labs.size(2) != self.lab_proj.in_features:
                logger.warning(f"实验室值特征维度不匹配: 输入 {labs.size(2)}, 预期 {self.lab_proj.in_features}")
                valid_dim = min(labs.size(2), self.lab_proj.in_features)
                fixed_labs = torch.zeros(batch_size, seq_len, self.lab_proj.in_features, device=labs.device)
                fixed_labs[:, :, :valid_dim] = labs[:, :, :valid_dim]
                labs = fixed_labs
                
            if drugs.size(2) != self.drug_proj.in_features:
                logger.warning(f"药物使用特征维度不匹配: 输入 {drugs.size(2)}, 预期 {self.drug_proj.in_features}")
                valid_dim = min(drugs.size(2), self.drug_proj.in_features)
                fixed_drugs = torch.zeros(batch_size, seq_len, self.drug_proj.in_features, device=drugs.device)
                fixed_drugs[:, :, :valid_dim] = drugs[:, :, :valid_dim]
                drugs = fixed_drugs
                
            if text_embed.size(1) != self.text_proj.in_features:
                logger.warning(f"文本嵌入维度不匹配: 输入 {text_embed.size(1)}, 预期 {self.text_proj.in_features}")
                valid_dim = min(text_embed.size(1), self.text_proj.in_features)
                fixed_text = torch.zeros(batch_size, self.text_proj.in_features, device=text_embed.device)
                fixed_text[:, :valid_dim] = text_embed[:, :valid_dim]
                text_embed = fixed_text
                
            if kg_embed.size(1) != self.kg_proj.in_features:
                logger.warning(f"知识图谱嵌入维度不匹配: 输入 {kg_embed.size(1)}, 预期 {self.kg_proj.in_features}")
                # 不仅调整大小，而是创建一个完全适应预期维度的新嵌入
                # 如果输入维度大于预期维度，保留最重要的前N个维度
                # 如果输入维度小于预期维度，通过线性插值扩展至所需维度
                if kg_embed.size(1) < self.kg_proj.in_features:
                    # 输入维度小于预期，扩展嵌入
                    expand_factor = self.kg_proj.in_features // kg_embed.size(1) + 1
                    repeated_embed = kg_embed.repeat(1, expand_factor)
                    fixed_kg = repeated_embed[:, :self.kg_proj.in_features]
                else:
                    # 输入维度大于预期，保留前N个维度
                    fixed_kg = kg_embed[:, :self.kg_proj.in_features]
                
                # 使用随机噪声填充剩余维度（如果有）
                if fixed_kg.size(1) < self.kg_proj.in_features:
                    padding_size = self.kg_proj.in_features - fixed_kg.size(1)
                    padding = torch.randn(batch_size, padding_size, device=kg_embed.device) * 0.01
                    fixed_kg = torch.cat([fixed_kg, padding], dim=1)
                
                kg_embed = fixed_kg
            
            # 生命体征特征处理
            vitals_proj = self.vitals_proj(vitals)  # [batch, seq_len, hidden_dim]
            vitals_proj = torch.nan_to_num(vitals_proj, nan=0.0, posinf=1.0, neginf=-1.0)
            vitals_proj = self.vitals_norm(vitals_proj)
            
            # 实验室值特征处理
            labs_proj = self.lab_proj(labs)        # [batch, seq_len, hidden_dim]
            labs_proj = torch.nan_to_num(labs_proj, nan=0.0, posinf=1.0, neginf=-1.0)
            labs_proj = self.lab_norm(labs_proj)
            
            # 药物使用特征处理
            drugs_proj = self.drug_proj(drugs)      # [batch, seq_len, hidden_dim]
            drugs_proj = torch.nan_to_num(drugs_proj, nan=0.0, posinf=1.0, neginf=-1.0)
            drugs_proj = self.drug_norm(drugs_proj)
            
            # 文本嵌入处理
            text_proj = self.text_proj(text_embed)  # [batch, hidden_dim]
            text_proj = torch.nan_to_num(text_proj, nan=0.0, posinf=1.0, neginf=-1.0)
            text_proj = self.text_norm(text_proj)
            text_proj = text_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim]
            
            # 知识图谱嵌入处理
            kg_proj = self.kg_proj(kg_embed)      # [batch, hidden_dim]
            kg_proj = torch.nan_to_num(kg_proj, nan=0.0, posinf=1.0, neginf=-1.0)
            kg_proj = self.kg_norm(kg_proj)
            kg_proj = kg_proj.unsqueeze(1).expand(-1, seq_len, -1)      # [batch, seq_len, hidden_dim]
        except Exception as e:
            logger.error(f"特征投影错误: {e}")
            logger.error(f"输入维度: vitals={vitals.shape}, labs={labs.shape}, drugs={drugs.shape}, text={text_embed.shape}, kg={kg_embed.shape}")
            logger.error(f"投影层维度: vitals_proj=({self.vitals_proj.in_features}, {self.vitals_proj.out_features}), "
                         f"lab_proj=({self.lab_proj.in_features}, {self.lab_proj.out_features}), "
                         f"drug_proj=({self.drug_proj.in_features}, {self.drug_proj.out_features}), "
                         f"text_proj=({self.text_proj.in_features}, {self.text_proj.out_features}), "
                         f"kg_proj=({self.kg_proj.in_features}, {self.kg_proj.out_features})")
            
            # 创建全零特征作为回退
            device = vitals.device
            vitals_proj = torch.zeros(batch_size, seq_len, self.vitals_norm.normalized_shape[0], device=device)
            labs_proj = torch.zeros_like(vitals_proj)
            drugs_proj = torch.zeros_like(vitals_proj)
            text_proj = torch.zeros_like(vitals_proj)
            kg_proj = torch.zeros_like(vitals_proj)
        
        # 特征融合 - 使用加权和而不是简单相加
        alpha = 0.2  # 各模态权重
        fused_features = (
            vitals_proj * alpha + 
            labs_proj * alpha + 
            drugs_proj * alpha + 
            text_proj * alpha + 
            kg_proj * alpha
        )
        
        # 融合后归一化
        fused_features = self.fusion_norm(fused_features)
        
        # 检查融合特征是否有NaN或Inf
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            logger.warning("融合特征包含NaN或Inf值，进行替换")
            fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 添加位置编码
        encoded_features = self.pos_encoder(fused_features)
        
        # 添加时间嵌入
        try:
            # 限制时间索引范围
            safe_time_indices = torch.clamp(time_indices.long(), 0, 4999)
            time_embed = self.time_embedding(safe_time_indices)
            time_embed = torch.nan_to_num(time_embed, nan=0.0, posinf=1.0, neginf=-1.0)
            encoded_features = encoded_features + time_embed
        except Exception as e:
            logger.error(f"时间嵌入错误: {e}")
            # 不添加时间嵌入
        
        # 创建注意力掩码（如果未提供）
        if attention_mask is None:
            # 使用更宽松的掩码条件
            vitals_valid = vitals.sum(dim=2) != 0
            labs_valid = labs.sum(dim=2) != 0
            drugs_valid = drugs.sum(dim=2) != 0
            valid_positions = vitals_valid | labs_valid | drugs_valid
            
            # 确保特征非零，如果所有特征都是零，添加一些随机特征
            if not valid_positions.any():
                logger.error("注意力权重计算中所有特征都是零")
                raise ValueError("无法计算注意力权重：所有特征都是零")
            
            attention_mask = ~valid_positions
        
        # 确保所有序列至少有一个非掩码位置
        if attention_mask.all(dim=1).any():
            # 找出所有位置都被掩码的样本
            fully_masked = attention_mask.all(dim=1)
            num_fully_masked = fully_masked.sum().item()
            if num_fully_masked > 0:
                logger.error(f"注意力权重计算中发现{num_fully_masked}个样本的所有位置都被掩码")
                raise ValueError("无法计算注意力权重：样本被完全掩码")
        
        # 将掩码转换为布尔类型，确保其维度正确
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # 检查是否有有效的嵌入特征
        if torch.isnan(encoded_features).any() or torch.isinf(encoded_features).any():
            logger.error("编码特征包含NaN或Inf值")
            # 替换NaN和Inf值
            encoded_features = torch.nan_to_num(encoded_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Transformer编码
        try:
            # 为了稳定性，将encoder_features的值限制在合理范围内
            encoded_features = torch.clamp(encoded_features, -10.0, 10.0)
            transformer_output = self.transformer_encoder(encoded_features, src_key_padding_mask=attention_mask)
            
            # 检查transformer输出是否有NaN或Inf
            if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
                logger.warning("Transformer输出包含NaN或Inf值，进行替换")
                transformer_output = torch.nan_to_num(transformer_output, nan=0.0, posinf=1.0, neginf=-1.0)
        except RuntimeError as e:
            logger.error(f"Transformer编码错误: {e}")
            logger.info(f"编码特征形状: {encoded_features.shape}, 注意力掩码形状: {attention_mask.shape}")
            logger.info(f"注意力掩码中True的比例: {attention_mask.float().mean().item()}")
            # 使用编码特征作为回退
            transformer_output = encoded_features
        
        # 限制transformer输出的范围
        transformer_output = torch.clamp(transformer_output, -10.0, 10.0)
        
        # 生成每个时间点的风险预测
        try:
            risk_scores = self.output_layer(transformer_output)
            
            # 确保风险分数在[0,1]范围内
            risk_scores = torch.clamp(risk_scores, 0.0, 1.0)
        except Exception as e:
            logger.error(f"输出层错误: {e}")
            # 创建默认风险分数（0.5）
            risk_scores = torch.ones(batch_size, seq_len, 1, device=vitals.device) * 0.5
        
        return risk_scores.squeeze(-1)  # 返回 [batch, seq_len] 形状的风险分数
    
    def get_attention_weights(self, vitals, labs, drugs, text_embed, kg_embed, time_indices, attention_mask=None):
        """
        获取注意力权重，用于可解释性分析
        
        Args:
            与forward方法参数相同
            
        Returns:
            list: 每层的注意力权重列表
        """
        # 获取批次大小和序列长度
        batch_size, seq_len = vitals.size(0), vitals.size(1)
        
        # 确保序列长度不为零
        if seq_len == 0:
            logger.warning("输入序列长度为零，无法计算注意力权重")
            return []
        
        # 处理输入特征中的NaN和Inf值
        vitals = torch.nan_to_num(vitals, nan=0.0, posinf=1.0, neginf=-1.0)
        labs = torch.nan_to_num(labs, nan=0.0, posinf=1.0, neginf=-1.0)
        drugs = torch.nan_to_num(drugs, nan=0.0, posinf=1.0, neginf=-1.0)
        text_embed = torch.nan_to_num(text_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        kg_embed = torch.nan_to_num(kg_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 投影并归一化各模态特征
        vitals_proj = self.vitals_proj(vitals)
        vitals_proj = torch.nan_to_num(vitals_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        vitals_proj = self.vitals_norm(vitals_proj)
        
        labs_proj = self.lab_proj(labs)
        labs_proj = torch.nan_to_num(labs_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        labs_proj = self.lab_norm(labs_proj)
        
        drugs_proj = self.drug_proj(drugs)
        drugs_proj = torch.nan_to_num(drugs_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        drugs_proj = self.drug_norm(drugs_proj)
        
        text_proj = self.text_proj(text_embed)
        text_proj = torch.nan_to_num(text_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        text_proj = self.text_norm(text_proj)
        text_proj = text_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        kg_proj = self.kg_proj(kg_embed)
        kg_proj = torch.nan_to_num(kg_proj, nan=0.0, posinf=1.0, neginf=-1.0)
        kg_proj = self.kg_norm(kg_proj)
        kg_proj = kg_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 特征融合
        alpha = 0.2  # 各模态权重
        fused_features = (
            vitals_proj * alpha + 
            labs_proj * alpha + 
            drugs_proj * alpha + 
            text_proj * alpha + 
            kg_proj * alpha
        )
        
        # 融合后归一化
        fused_features = self.fusion_norm(fused_features)
        fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 添加位置编码
        encoded_features = self.pos_encoder(fused_features)
        
        # 添加时间嵌入
        safe_time_indices = torch.clamp(time_indices.long(), 0, 4999)
        time_embed = self.time_embedding(safe_time_indices)
        time_embed = torch.nan_to_num(time_embed, nan=0.0, posinf=1.0, neginf=-1.0)
        encoded_features = encoded_features + time_embed
        
        # 创建注意力掩码（如果未提供）
        if attention_mask is None:
            # 使用更宽松的掩码条件
            vitals_valid = vitals.sum(dim=2) != 0
            labs_valid = labs.sum(dim=2) != 0
            drugs_valid = drugs.sum(dim=2) != 0
            valid_positions = vitals_valid | labs_valid | drugs_valid
            
            # 确保特征非零，如果所有特征都是零，添加一些随机特征
            if not valid_positions.any():
                logger.error("注意力权重计算中所有特征都是零")
                raise ValueError("无法计算注意力权重：所有特征都是零")
            
            attention_mask = ~valid_positions
        
        # 确保所有序列至少有一个非掩码位置
        if attention_mask.all(dim=1).any():
            # 找出所有位置都被掩码的样本
            fully_masked = attention_mask.all(dim=1)
            num_fully_masked = fully_masked.sum().item()
            if num_fully_masked > 0:
                logger.error(f"注意力权重计算中发现{num_fully_masked}个样本的所有位置都被掩码")
                raise ValueError("无法计算注意力权重：样本被完全掩码")
        
        # 将掩码转换为布尔类型
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # 收集每层的注意力权重
        attention_weights = []
        
        # 手动逐层处理，以收集注意力权重
        x = encoded_features
        for layer in self.transformer_encoder.layers:
            try:
                # 自注意力计算
                q = layer.self_attn.q_proj(x)
                k = layer.self_attn.k_proj(x)
                v = layer.self_attn.v_proj(x)
                
                # 检查并处理NaN和Inf值
                q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
                k = torch.nan_to_num(k, nan=0.0, posinf=1.0, neginf=-1.0)
                v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 重塑为多头形式
                head_dim = layer.self_attn.head_dim
                num_heads = layer.self_attn.num_heads
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                
                # 计算注意力分数
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                
                # 限制分数范围，防止数值不稳定
                scores = torch.clamp(scores, -10.0, 10.0)
                
                # 应用掩码
                if attention_mask is not None:
                    scores = scores.masked_fill(
                        attention_mask.unsqueeze(1).unsqueeze(2),
                        float('-inf')
                    )
                
                # 应用softmax
                attn_weights = F.softmax(scores, dim=-1)
                
                # 处理可能的NaN (如果所有值都是-inf)
                attn_weights = torch.nan_to_num(attn_weights, nan=1.0/seq_len)
                
                # 收集注意力权重
                attention_weights.append(attn_weights)
                
                # 继续前向传播
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
                
                x = layer.norm1(x + layer.dropout1(attn_output))
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                ff_output = torch.nan_to_num(ff_output, nan=0.0, posinf=1.0, neginf=-1.0)
                
                x = layer.norm2(x + layer.dropout2(ff_output))
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception as e:
                logger.error(f"计算注意力权重时出错: {e}")
                break
        
        return attention_weights 