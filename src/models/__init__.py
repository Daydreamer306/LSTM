"""模型定义模块

包含LSTM模块、Transformer模块和融合模型
"""

from .lstm_module import (
    LSTMEncoder, 
    ResidualLSTM, 
    MultiScaleLSTM
)

from .transformer_module import (
    PositionalEncoding,
    MultiHeadAttention, 
    TransformerEncoderLayer,
    TransformerEncoder,
    TemporalTransformer
)

from .fusion_model import (
    LSTMTransformerFusion,
    AdvancedLSTMTransformerFusion,
    create_model
)

from .model_utils import (
    ModelFactory,
    ModelManager,
    validate_model_config
)

__all__ = [
    # LSTM模块
    'LSTMEncoder',
    'ResidualLSTM', 
    'MultiScaleLSTM',
    
    # Transformer模块
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerEncoderLayer', 
    'TransformerEncoder',
    'TemporalTransformer',
    
    # 融合模型
    'LSTMTransformerFusion',
    'AdvancedLSTMTransformerFusion',
    'create_model',
    
    # 工具函数
    'ModelFactory',
    'ModelManager', 
    'validate_model_config'
]
