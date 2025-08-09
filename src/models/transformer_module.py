"""
Transformer模块定义
用于捕捉时序数据的全局依赖关系和注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        位置编码初始化
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout比例
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除法项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        # 注册为buffer，不参与梯度计算
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 (seq_len, batch_size, d_model) 或 (batch_size, seq_len, d_model)
            
        Returns:
            加入位置编码后的张量
        """
        # 如果输入是 (batch_size, seq_len, d_model)，转换为 (seq_len, batch_size, d_model)
        if x.dim() == 3 and x.size(1) > x.size(0):
            x = x.transpose(0, 1)
            pe_added = x + self.pe[:x.size(0), :]
            return self.dropout(pe_added.transpose(0, 1))
        else:
            pe_added = x + self.pe[:x.size(0), :]
            return self.dropout(pe_added)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        """
        多头注意力初始化
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: dropout比例
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力
        
        Args:
            Q: Query张量 (batch_size, nhead, seq_len, d_k)
            K: Key张量 (batch_size, nhead, seq_len, d_k)
            V: Value张量 (batch_size, nhead, seq_len, d_k)
            mask: 注意力掩码
            
        Returns:
            output: 注意力输出 (batch_size, nhead, seq_len, d_k)
            attention: 注意力权重 (batch_size, nhead, seq_len, seq_len)
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax归一化
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力权重
        output = torch.matmul(attention, V)
        
        return output, attention
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多头注意力前向传播
        
        Args:
            query: Query张量 (batch_size, seq_len, d_model)
            key: Key张量 (batch_size, seq_len, d_model)
            value: Value张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            
        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
            attention: 注意力权重 (batch_size, nhead, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # 线性变换
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)
        
        # 缩放点积注意力
        attn_output, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 连接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attn_output)
        
        return output, attention


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Transformer编码器层初始化
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            activation: 激活函数类型
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器层前向传播
        
        Args:
            src: 输入张量 (batch_size, seq_len, d_model)
            src_mask: 源序列掩码
            
        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_output))
        
        return src


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_len: int = 5000):
        """
        Transformer编码器初始化
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            activation: 激活函数类型
            max_len: 最大序列长度
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播
        
        Args:
            src: 输入张量 (batch_size, seq_len, d_model)
            src_mask: 源序列掩码
            
        Returns:
            output: 编码器输出 (batch_size, seq_len, d_model)
        """
        # 位置编码
        src = self.pos_encoding(src)
        
        # 通过编码器层
        for layer in self.layers:
            src = layer(src, src_mask)
        
        # 最终层归一化
        src = self.norm(src)
        
        return src


class TemporalTransformer(nn.Module):
    """专为时序数据设计的Transformer"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1, activation: str = "gelu"):
        """
        时序Transformer初始化
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            activation: 激活函数类型
        """
        super(TemporalTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """创建填充掩码"""
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_temporal_mask(self, seq_len: int, lookback_window: Optional[int] = None) -> torch.Tensor:
        """创建时序掩码，限制注意力的时间窗口"""
        mask = torch.ones(seq_len, seq_len)
        
        if lookback_window is not None:
            # 创建滑动窗口掩码
            for i in range(seq_len):
                start = max(0, i - lookback_window)
                mask[i, :start] = 0
        
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时序Transformer前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            mask: 注意力掩码
            
        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影到模型维度
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x, mask)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x


def test_transformer_modules():
    """测试Transformer模块"""
    batch_size, seq_len, input_size = 4, 60, 128
    d_model, nhead = 256, 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试位置编码
    print("测试位置编码...")
    pe = PositionalEncoding(d_model)
    x_proj = torch.randn(batch_size, seq_len, d_model)
    pe_output = pe(x_proj)
    print(f"位置编码输出形状: {pe_output.shape}")
    
    # 测试多头注意力
    print("测试多头注意力...")
    mha = MultiHeadAttention(d_model, nhead)
    attn_output, attention = mha(x_proj, x_proj, x_proj)
    print(f"多头注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attention.shape}")
    
    # 测试Transformer编码器
    print("测试Transformer编码器...")
    transformer = TransformerEncoder(d_model, nhead, num_layers=2)
    transformer_output = transformer(x_proj)
    print(f"Transformer编码器输出形状: {transformer_output.shape}")
    
    # 测试时序Transformer
    print("测试时序Transformer...")
    temporal_transformer = TemporalTransformer(input_size, d_model, nhead)
    output = temporal_transformer(x)
    print(f"时序Transformer输出形状: {output.shape}")


if __name__ == "__main__":
    test_transformer_modules()
