import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Chomp1d(nn.Module):
    """
    对于因果卷积，移除右侧填充
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    TCN的基本构建块，包含两个因果卷积层、非线性激活函数、标准化层、dropout和残差连接
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, 
            padding=padding, dilation=dilation
        ))
        
        # 移除多余的填充，确保因果性
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, 
            padding=padding, dilation=dilation
        ))
        
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 如果输入输出通道数不同，添加1x1卷积进行匹配
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # 模块初始化
        self.init_weights()
        
    def init_weights(self):
        """
        初始化权重，提高训练稳定性
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, n_inputs, seq_len]
            
        返回:
            output: 输出张量 [batch_size, n_outputs, seq_len]
        """
        # 第一个卷积块
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    时间卷积网络(TCN)
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        参数:
            num_inputs: 输入特征维度
            num_channels: 每层通道数的列表，长度确定层数，值确定通道数
            kernel_size: 卷积核大小
            dropout: Dropout比例
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长的空洞率
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, num_inputs]
            
        返回:
            输出序列 [batch_size, seq_len, num_channels[-1]]
        """
        # 调整输入维度从[batch, seq_len, features]到[batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 应用TCN
        y = self.network(x)
        
        # 恢复维度为[batch, seq_len, features]
        return y.transpose(1, 2)

class SepsisTCNModel(nn.Module):
    """
    基于TCN的脓毒症预测模型
    """
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 512], 
                 kernel_size=3, dropout=0.2, output_attention=True):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dims: 每层的通道数
            kernel_size: 卷积核大小
            dropout: Dropout比例
            output_attention: 是否使用输出注意力机制
        """
        super(SepsisTCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.output_attention = output_attention
        final_dim = hidden_dims[-1]
        
        # 如果使用注意力，添加注意力层
        if output_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(final_dim, final_dim // 2),
                nn.Tanh(),
                nn.Linear(final_dim // 2, 1)
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def attention(self, features, mask=None):
        """
        计算注意力权重
        
        参数:
            features: 特征 [batch_size, seq_len, hidden_dim]
            mask: 掩码 [batch_size, seq_len]
            
        返回:
            加权特征 [batch_size, hidden_dim]
        """
        # 计算注意力分数 [batch_size, seq_len, 1]
        scores = self.attention_layer(features)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得权重
        weights = F.softmax(scores, dim=1)
        
        # 计算加权和
        return torch.sum(features * weights, dim=1)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, input_dim]
            mask: 序列掩码 [batch_size, seq_len]
            
        返回:
            每个时间点的风险分数 [batch_size, seq_len, 1]
        """
        # 应用TCN
        tcn_output = self.tcn(x)  # [batch_size, seq_len, hidden_dim]
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配tcn_output
            mask_expanded = mask.unsqueeze(-1).expand_as(tcn_output)
            tcn_output = tcn_output * mask_expanded
        
        if self.output_attention:
            # 使用注意力机制计算每个时间点的加权输出
            outputs = []
            for t in range(tcn_output.size(1)):
                # 获取到当前时间点t的所有输出
                current_features = tcn_output[:, :t+1, :]
                current_mask = mask[:, :t+1] if mask is not None else None
                
                if t == 0:  # 只有一个时间点时，直接使用
                    outputs.append(current_features.squeeze(1))
                else:  # 多个时间点时，使用注意力
                    outputs.append(self.attention(current_features, current_mask))
            
            # 堆叠所有时间点的输出
            combined_output = torch.stack(outputs, dim=1)
        else:
            # 不使用注意力，直接使用TCN输出
            combined_output = tcn_output
        
        # 应用输出层
        risk_scores = self.output_layer(combined_output)
        
        return risk_scores 