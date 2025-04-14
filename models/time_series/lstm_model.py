import torch
import torch.nn as nn

class SepsisLSTMModel(nn.Module):
    """
    基于LSTM的脓毒症时序预测模型
    
    支持多层双向LSTM、残差连接和注意力机制
    用于捕捉患者生命体征和实验室检查数据的时间动态变化
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, bidirectional=True, 
                 dropout=0.2, use_attention=True, use_residual=True):
        super(SepsisLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 计算LSTM输出维度
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 注意力层
        if use_attention:
            self.attention_weight = nn.Linear(self.lstm_output_dim, 1)
        
        # 输入投影层（用于残差连接）
        if use_residual:
            self.input_projection = nn.Linear(input_dim, self.lstm_output_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def attention(self, lstm_output):
        """
        计算注意力权重
        
        参数:
            lstm_output: LSTM输出 [batch_size, seq_len, lstm_output_dim]
            
        返回:
            加权后的LSTM输出表示 [batch_size, lstm_output_dim]
        """
        # 计算注意力分数 [batch_size, seq_len, 1]
        attn_scores = self.attention_weight(lstm_output)
        
        # 对seq_len维度应用softmax
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 计算加权和
        weighted_output = torch.sum(lstm_output * attn_weights, dim=1)
        
        return weighted_output
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, input_dim]
            mask: 序列掩码 [batch_size, seq_len]，表示哪些时间点是有效的
            
        返回:
            每个时间点的预测风险概率 [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM处理
        lstm_output, _ = self.lstm(x)  # [batch_size, seq_len, lstm_output_dim]
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码维度以匹配lstm_output
            extended_mask = mask.unsqueeze(-1).expand_as(lstm_output)
            # 将掩码位置设为0
            lstm_output = lstm_output * extended_mask
        
        # 残差连接
        if self.use_residual:
            residual = self.input_projection(x)
            lstm_output = lstm_output + residual
        
        # 输出处理
        if self.use_attention:
            # 使用注意力机制计算每个时间点的注意力权重
            time_outputs = []
            for t in range(seq_len):
                # 获取到当前时间点t的所有输出
                current_output = lstm_output[:, :t+1, :]
                if t == 0:  # 只有一个时间点时，直接使用
                    time_outputs.append(current_output.squeeze(1))
                else:  # 多个时间点时，使用注意力
                    time_outputs.append(self.attention(current_output))
            
            # 堆叠所有时间点的输出
            combined_output = torch.stack(time_outputs, dim=1)  # [batch_size, seq_len, lstm_output_dim]
        else:
            # 不使用注意力，直接使用LSTM输出
            combined_output = lstm_output
        
        # 计算每个时间点的风险概率
        risk_scores = self.output_layer(combined_output)  # [batch_size, seq_len, 1]
        
        return risk_scores

class TemporalDecayAttention(nn.Module):
    """
    时间衰减注意力机制
    
    考虑时间距离，给予近期数据更高的权重
    """
    def __init__(self, hidden_dim, decay_rate=0.1):
        super(TemporalDecayAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        
        # 注意力参数
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hidden_states, time_intervals=None):
        """
        应用时间衰减注意力
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            time_intervals: 时间间隔 [batch_size, seq_len]，从当前时间点到每个历史时间点的间隔
            
        返回:
            注意力加权的隐藏状态 [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # 计算查询、键和值
        q = self.query(hidden_states[:, -1, :]).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = self.key(hidden_states)  # [batch_size, seq_len, hidden_dim]
        v = self.value(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [batch_size, 1, seq_len]
        
        # 应用时间衰减
        if time_intervals is not None:
            # 计算衰减系数
            decay_factor = torch.exp(-self.decay_rate * time_intervals).unsqueeze(1)  # [batch_size, 1, seq_len]
            attn_scores = attn_scores * decay_factor
        
        # 应用softmax获得注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, 1, seq_len]
        
        # 计算注意力输出
        output = torch.matmul(attn_weights, v)  # [batch_size, 1, hidden_dim]
        
        return output.squeeze(1)  # [batch_size, hidden_dim] 