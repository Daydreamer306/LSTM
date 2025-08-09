"""
LSTM模块定义
用于捕捉时序数据的局部长短期依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LSTMEncoder(nn.Module):
    """LSTM编码器模块"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        batch_first: bool = True
    ):
        """
        初始化LSTM编码器
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏状态维度
            num_layers: LSTM层数
            dropout: dropout比例
            bidirectional: 是否双向LSTM
            batch_first: 是否batch维度在前
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # 计算LSTM输出维度
        self.lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.lstm_output_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            hidden: 初始隐藏状态，None表示使用零初始化
            
        Returns:
            output: LSTM输出 (batch_size, seq_len, hidden_size * directions)
            hidden: 最终隐藏状态
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 应用dropout和层归一化
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态"""
        directions = 2 if self.bidirectional else 1
        
        h_0 = torch.zeros(
            self.num_layers * directions, 
            batch_size, 
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * directions, 
            batch_size, 
            self.hidden_size,
            device=device
        )
        
        return h_0, c_0


class ResidualLSTM(nn.Module):
    """带残差连接的LSTM模块"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(ResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # LSTM编码器
        self.lstm_encoder = LSTMEncoder(
            input_size, hidden_size, num_layers, dropout, bidirectional
        )
        
        # 投影层，用于维度匹配
        if input_size != self.lstm_output_size:
            self.projection = nn.Linear(input_size, self.lstm_output_size)
        else:
            self.projection = nn.Identity()
        
        # 残差连接的门控机制
        self.gate = nn.Sequential(
            nn.Linear(self.lstm_output_size + input_size, self.lstm_output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        带残差连接的前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            
        Returns:
            output: 输出张量 (batch_size, seq_len, lstm_output_size)
        """
        # LSTM编码
        lstm_out, _ = self.lstm_encoder(x)
        
        # 输入投影
        x_proj = self.projection(x)
        
        # 门控残差连接
        gate_input = torch.cat([lstm_out, x], dim=-1)
        gate = self.gate(gate_input)
        
        # 残差连接
        output = gate * lstm_out + (1 - gate) * x_proj
        
        return output


class MultiScaleLSTM(nn.Module):
    """多尺度LSTM模块，处理不同时间尺度的依赖关系"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [64, 128, 64],
        scales: list = [1, 2, 4],
        dropout: float = 0.2
    ):
        """
        多尺度LSTM初始化
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 各尺度LSTM的隐藏维度
            scales: 时间尺度列表（步长）
            dropout: dropout比例
        """
        super(MultiScaleLSTM, self).__init__()
        
        assert len(hidden_sizes) == len(scales), "隐藏维度数量必须与尺度数量相等"
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.scales = scales
        self.num_scales = len(scales)
        
        # 各尺度的LSTM
        self.lstm_layers = nn.ModuleList()
        for i, (hidden_size, scale) in enumerate(zip(hidden_sizes, scales)):
            lstm = LSTMEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                bidirectional=True
            )
            self.lstm_layers.append(lstm)
        
        # 特征融合层
        total_output_size = sum(h * 2 for h in hidden_sizes)  # 双向LSTM
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_output_size, total_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_output_size // 2, total_output_size // 4),
            nn.LayerNorm(total_output_size // 4)
        )
        
        self.output_size = total_output_size // 4
        
    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """下采样到指定尺度"""
        if scale == 1:
            return x
        
        batch_size, seq_len, features = x.shape
        # 使用平均池化进行下采样
        if seq_len % scale != 0:
            # 填充到可以整除的长度
            pad_len = scale - (seq_len % scale)
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            seq_len += pad_len
        
        x_reshaped = x.view(batch_size, seq_len // scale, scale, features)
        x_downsampled = x_reshaped.mean(dim=2)
        
        return x_downsampled
    
    def _upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """上采样到目标长度"""
        batch_size, seq_len, features = x.shape
        
        if seq_len == target_len:
            return x
        
        # 使用插值进行上采样
        x_permuted = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x_upsampled = F.interpolate(
            x_permuted, size=target_len, mode='linear', align_corners=True
        )
        x_upsampled = x_upsampled.permute(0, 2, 1)  # (batch, seq_len, features)
        
        return x_upsampled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度LSTM前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            
        Returns:
            output: 融合后的输出 (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # 各尺度特征提取
        scale_outputs = []
        for lstm, scale in zip(self.lstm_layers, self.scales):
            # 下采样
            x_scaled = self._downsample(x, scale)
            
            # LSTM编码
            lstm_out, _ = lstm(x_scaled)
            
            # 上采样回原始长度
            lstm_out_upsampled = self._upsample(lstm_out, seq_len)
            
            scale_outputs.append(lstm_out_upsampled)
        
        # 特征融合
        fused_features = torch.cat(scale_outputs, dim=-1)
        output = self.feature_fusion(fused_features)
        
        return output


def test_lstm_modules():
    """测试LSTM模块"""
    batch_size, seq_len, input_size = 4, 60, 10
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试基础LSTM编码器
    print("测试LSTM编码器...")
    lstm_encoder = LSTMEncoder(input_size, hidden_size=64, num_layers=2)
    output, hidden = lstm_encoder(x)
    print(f"LSTM编码器输出形状: {output.shape}")
    
    # 测试残差LSTM
    print("测试残差LSTM...")
    res_lstm = ResidualLSTM(input_size, hidden_size=64)
    output = res_lstm(x)
    print(f"残差LSTM输出形状: {output.shape}")
    
    # 测试多尺度LSTM
    print("测试多尺度LSTM...")
    ms_lstm = MultiScaleLSTM(input_size, hidden_sizes=[32, 64, 32])
    output = ms_lstm(x)
    print(f"多尺度LSTM输出形状: {output.shape}")


if __name__ == "__main__":
    test_lstm_modules()
