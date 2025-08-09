"""
LSTM + Transformer 融合模型
结合LSTM的局部时序建模能力和Transformer的全局依赖捕捉能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from .lstm_module import LSTMEncoder, ResidualLSTM, MultiScaleLSTM
from .transformer_module import TemporalTransformer, TransformerEncoder


class LSTMTransformerFusion(nn.Module):
    """LSTM + Transformer 融合模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        融合模型初始化
        
        Args:
            config: 模型配置字典
        """
        super(LSTMTransformerFusion, self).__init__()
        
        self.config = config
        model_config = config['model']
        lstm_config = model_config['lstm']
        transformer_config = model_config['transformer']
        pred_config = model_config['prediction_head']
        
        # 基本参数
        self.input_size = lstm_config['input_size']
        self.lstm_hidden_size = lstm_config['hidden_size']
        self.lstm_num_layers = lstm_config['num_layers']
        self.lstm_dropout = lstm_config['dropout']
        self.bidirectional = lstm_config['bidirectional']
        
        self.d_model = transformer_config['d_model']
        self.nhead = transformer_config['nhead']
        self.transformer_num_layers = transformer_config['num_layers']
        self.dim_feedforward = transformer_config['dim_feedforward']
        self.transformer_dropout = transformer_config['dropout']
        self.activation = transformer_config['activation']
        
        self.prediction_length = config['data']['prediction_length']
        self.target_dim = pred_config['output_dim']
        
        # 计算LSTM输出维度
        self.lstm_output_size = self.lstm_hidden_size * (2 if self.bidirectional else 1)
        
        # LSTM编码器
        self.lstm_encoder = LSTMEncoder(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout,
            bidirectional=self.bidirectional
        )
        
        # LSTM输出到Transformer输入的维度适配
        if self.lstm_output_size != self.d_model:
            self.lstm_to_transformer = nn.Linear(self.lstm_output_size, self.d_model)
        else:
            self.lstm_to_transformer = nn.Identity()
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.transformer_num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.activation
        )
        
        # 预测头
        self.prediction_head = self._build_prediction_head(pred_config)
        
        # 初始化权重
        self._init_weights()
    
    def _build_prediction_head(self, pred_config: Dict[str, Any]) -> nn.Module:
        """构建预测头"""
        hidden_dims = pred_config['hidden_dims']
        dropout = pred_config['dropout']
        
        layers = []
        input_dim = self.d_model
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # 输出层：生成未来prediction_length步的预测
        layers.append(nn.Linear(input_dim, self.prediction_length * self.target_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier均匀初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.LSTM):
                # LSTM权重初始化
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            predictions: 预测结果 (batch_size, prediction_length, target_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. LSTM编码 - 捕捉局部时序依赖
        lstm_output, _ = self.lstm_encoder(x)  # (batch_size, seq_len, lstm_output_size)
        
        # 2. 维度适配到Transformer
        transformer_input = self.lstm_to_transformer(lstm_output)  # (batch_size, seq_len, d_model)
        
        # 3. Transformer编码 - 捕捉全局依赖和自注意力
        transformer_output = self.transformer_encoder(transformer_input)  # (batch_size, seq_len, d_model)
        
        # 4. 聚合时序特征（使用最后几个时间步的平均）
        # 可以使用不同的聚合策略：最后一步、平均、注意力加权等
        aggregated_features = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # 5. 预测头 - 生成未来预测
        predictions_flat = self.prediction_head(aggregated_features)  # (batch_size, prediction_length * target_dim)
        
        # 6. 重塑为预测格式
        predictions = predictions_flat.view(batch_size, self.prediction_length, self.target_dim)
        
        return predictions


class AdvancedLSTMTransformerFusion(nn.Module):
    """高级LSTM + Transformer融合模型，包含更多融合策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super(AdvancedLSTMTransformerFusion, self).__init__()
        
        self.config = config
        model_config = config['model']
        lstm_config = model_config['lstm']
        transformer_config = model_config['transformer']
        pred_config = model_config['prediction_head']
        
        # 基本参数
        self.input_size = lstm_config['input_size']
        self.lstm_hidden_size = lstm_config['hidden_size']
        self.d_model = transformer_config['d_model']
        self.prediction_length = config['data']['prediction_length']
        self.target_dim = pred_config['output_dim']
        
        # 计算维度
        self.lstm_output_size = self.lstm_hidden_size * (2 if lstm_config['bidirectional'] else 1)
        
        # 多尺度LSTM
        self.multi_scale_lstm = MultiScaleLSTM(
            input_size=self.input_size,
            hidden_sizes=[64, 128, 64],
            scales=[1, 2, 4],
            dropout=lstm_config['dropout']
        )
        
        # 残差LSTM
        self.residual_lstm = ResidualLSTM(
            input_size=self.multi_scale_lstm.output_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            bidirectional=lstm_config['bidirectional']
        )
        
        # 时序Transformer
        self.temporal_transformer = TemporalTransformer(
            input_size=self.lstm_output_size,
            d_model=self.d_model,
            nhead=transformer_config['nhead'],
            num_layers=transformer_config['num_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout'],
            activation=transformer_config['activation']
        )
        
        # 特征融合层
        self.feature_fusion = nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(self.lstm_output_size + self.d_model, self.d_model),
                nn.Sigmoid()
            ),
            'transform': nn.Sequential(
                nn.Linear(self.lstm_output_size + self.d_model, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model)
            )
        })
        
        # 注意力聚合
        self.attention_aggregation = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=transformer_config['nhead'],
            dropout=transformer_config['dropout'],
            batch_first=True
        )
        
        # 预测头
        self.prediction_head = self._build_advanced_prediction_head(pred_config)
        
        self._init_weights()
    
    def _build_advanced_prediction_head(self, pred_config: Dict[str, Any]) -> nn.Module:
        """构建高级预测头"""
        hidden_dims = pred_config['hidden_dims']
        dropout = pred_config['dropout']
        
        # 多步预测头：为每个预测步长创建专用的预测分支
        prediction_heads = nn.ModuleList()
        
        for i in range(self.prediction_length):
            head_layers = []
            input_dim = self.d_model
            
            for hidden_dim in hidden_dims:
                head_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
            
            # 输出层
            head_layers.append(nn.Linear(input_dim, self.target_dim))
            
            prediction_heads.append(nn.Sequential(*head_layers))
        
        return prediction_heads
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高级融合前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            predictions: 预测结果 (batch_size, prediction_length, target_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 多尺度LSTM特征提取
        ms_lstm_output = self.multi_scale_lstm(x)  # (batch_size, seq_len, ms_output_size)
        
        # 2. 残差LSTM进一步编码
        lstm_output = self.residual_lstm(ms_lstm_output)  # (batch_size, seq_len, lstm_output_size)
        
        # 3. Transformer编码
        transformer_output = self.temporal_transformer(lstm_output)  # (batch_size, seq_len, d_model)
        
        # 4. 特征融合：门控机制结合LSTM和Transformer特征
        # 将LSTM输出维度调整到d_model
        lstm_projected = F.adaptive_avg_pool1d(
            lstm_output.transpose(1, 2), self.d_model
        ).transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # 拼接特征
        fused_features = torch.cat([lstm_projected, transformer_output], dim=-1)
        
        # 门控融合
        gate = self.feature_fusion['gate'](fused_features)
        fused_output = self.feature_fusion['transform'](fused_features)
        weighted_output = gate * transformer_output + (1 - gate) * lstm_projected
        
        # 5. 注意力聚合：将序列信息聚合为固定维度特征
        # 使用自注意力机制聚合时序特征
        aggregated_features, _ = self.attention_aggregation(
            weighted_output, weighted_output, weighted_output
        )
        
        # 取平均或使用最后时间步
        final_features = aggregated_features.mean(dim=1)  # (batch_size, d_model)
        
        # 6. 多步预测：为每个预测步长单独预测
        predictions = []
        for i, head in enumerate(self.prediction_head):
            step_prediction = head(final_features)  # (batch_size, target_dim)
            predictions.append(step_prediction.unsqueeze(1))
        
        # 拼接所有预测步长
        predictions = torch.cat(predictions, dim=1)  # (batch_size, prediction_length, target_dim)
        
        return predictions


def create_model(config: Dict[str, Any], use_advanced: bool = False) -> nn.Module:
    """
    创建模型实例
    
    Args:
        config: 配置字典
        use_advanced: 是否使用高级融合模型
        
    Returns:
        model: 模型实例
    """
    if use_advanced:
        return AdvancedLSTMTransformerFusion(config)
    else:
        return LSTMTransformerFusion(config)


def test_fusion_models():
    """测试融合模型"""
    # 创建测试配置
    config = {
        'data': {
            'sequence_length': 60,
            'prediction_length': 10
        },
        'model': {
            'lstm': {
                'input_size': 8,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'transformer': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'activation': 'gelu'
            },
            'prediction_head': {
                'hidden_dims': [128, 64],
                'dropout': 0.1,
                'output_dim': 2
            }
        }
    }
    
    batch_size, seq_len, input_size = 4, 60, 8
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 测试基础融合模型
    print("测试基础LSTM-Transformer融合模型...")
    basic_model = create_model(config, use_advanced=False)
    basic_output = basic_model(x)
    print(f"基础模型输出形状: {basic_output.shape}")
    
    # 测试高级融合模型
    print("测试高级LSTM-Transformer融合模型...")
    advanced_model = create_model(config, use_advanced=True)
    advanced_output = advanced_model(x)
    print(f"高级模型输出形状: {advanced_output.shape}")
    
    # 计算参数数量
    basic_params = sum(p.numel() for p in basic_model.parameters())
    advanced_params = sum(p.numel() for p in advanced_model.parameters())
    
    print(f"基础模型参数数量: {basic_params:,}")
    print(f"高级模型参数数量: {advanced_params:,}")


if __name__ == "__main__":
    test_fusion_models()
