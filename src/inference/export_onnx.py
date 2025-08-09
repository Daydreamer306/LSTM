"""
ONNX模型导出工具
支持PyTorch模型导出为ONNX格式，用于部署和推理优化
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import logging
from dataclasses import dataclass

from ..models.fusion_model import LSTMTransformerFusion
from ..models.model_utils import create_model


@dataclass
class ONNXExportConfig:
    """ONNX导出配置"""
    model_path: str
    output_path: str
    input_shape: Tuple[int, int, int]  # (batch_size, sequence_length, input_size)
    opset_version: int = 11
    do_constant_folding: bool = True
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    optimize: bool = True
    verify: bool = True


class ONNXExporter:
    """ONNX模型导出器"""
    
    def __init__(self, config: ONNXExportConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """加载PyTorch模型"""
        try:
            # 加载模型检查点
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # 创建模型
            if 'model_config' in checkpoint:
                model = create_model(checkpoint['model_config'])
            else:
                # 默认配置
                model_config = {
                    'input_size': self.config.input_shape[2],
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'num_heads': 8,
                    'sequence_length': self.config.input_shape[1]
                }
                model = create_model(model_config)
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.logger.info("PyTorch模型加载成功")
            return model
            
        except Exception as e:
            self.logger.error(f"加载PyTorch模型失败: {e}")
            raise
            
    def export_to_onnx(self, model: torch.nn.Module) -> str:
        """导出模型为ONNX格式"""
        try:
            # 创建示例输入
            batch_size, sequence_length, input_size = self.config.input_shape
            dummy_input = torch.randn(batch_size, sequence_length, input_size).to(self.device)
            
            # 设置动态轴
            dynamic_axes = self.config.dynamic_axes or {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            # 导出为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                self.config.output_path,
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.do_constant_folding,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            self.logger.info(f"ONNX模型导出成功: {self.config.output_path}")
            return self.config.output_path
            
        except Exception as e:
            self.logger.error(f"ONNX模型导出失败: {e}")
            raise
            
    def optimize_onnx_model(self, onnx_path: str) -> str:
        """优化ONNX模型"""
        if not self.config.optimize:
            return onnx_path
            
        try:
            import onnxoptimizer
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 优化模型
            optimized_model = onnxoptimizer.optimize(onnx_model)
            
            # 保存优化后的模型
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            self.logger.info(f"ONNX模型优化完成: {optimized_path}")
            return optimized_path
            
        except ImportError:
            self.logger.warning("onnxoptimizer未安装，跳过优化步骤")
            return onnx_path
        except Exception as e:
            self.logger.error(f"ONNX模型优化失败: {e}")
            return onnx_path
            
    def verify_onnx_model(
        self, 
        onnx_path: str, 
        pytorch_model: torch.nn.Module
    ) -> bool:
        """验证ONNX模型"""
        if not self.config.verify:
            return True
            
        try:
            # 创建测试输入
            batch_size, sequence_length, input_size = self.config.input_shape
            test_input = torch.randn(batch_size, sequence_length, input_size)
            
            # PyTorch预测
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input.to(self.device)).cpu().numpy()
            
            # ONNX预测
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # 比较结果
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            tolerance = 1e-4
            is_valid = max_diff < tolerance
            
            if is_valid:
                self.logger.info(f"ONNX模型验证通过 (最大差异: {max_diff:.6f})")
            else:
                self.logger.warning(f"ONNX模型验证失败 (最大差异: {max_diff:.6f})")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"ONNX模型验证失败: {e}")
            return False
            
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """获取ONNX模型信息"""
        try:
            # 加载模型
            onnx_model = onnx.load(onnx_path)
            
            # 获取输入信息
            inputs_info = []
            for inp in onnx_model.graph.input:
                shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                inputs_info.append({
                    'name': inp.name,
                    'shape': shape,
                    'type': inp.type.tensor_type.elem_type
                })
                
            # 获取输出信息
            outputs_info = []
            for out in onnx_model.graph.output:
                shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
                outputs_info.append({
                    'name': out.name,
                    'shape': shape,
                    'type': out.type.tensor_type.elem_type
                })
                
            # 获取模型统计信息
            model_info = {
                'file_path': onnx_path,
                'file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
                'opset_version': onnx_model.opset_import[0].version,
                'inputs': inputs_info,
                'outputs': outputs_info,
                'num_nodes': len(onnx_model.graph.node),
                'producer_name': onnx_model.producer_name,
                'producer_version': onnx_model.producer_version
            }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"获取ONNX模型信息失败: {e}")
            return {}
            
    def save_model_info(self, model_info: Dict[str, Any], output_dir: str):
        """保存模型信息"""
        info_path = Path(output_dir) / 'model_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        self.logger.info(f"模型信息已保存: {info_path}")
        
    def export(self) -> Dict[str, Any]:
        """执行完整的导出流程"""
        try:
            # 加载PyTorch模型
            pytorch_model = self.load_pytorch_model()
            
            # 导出为ONNX
            onnx_path = self.export_to_onnx(pytorch_model)
            
            # 优化模型
            optimized_path = self.optimize_onnx_model(onnx_path)
            
            # 验证模型
            is_valid = self.verify_onnx_model(optimized_path, pytorch_model)
            
            # 获取模型信息
            model_info = self.get_model_info(optimized_path)
            model_info['validation_passed'] = is_valid
            
            # 保存模型信息
            output_dir = Path(self.config.output_path).parent
            self.save_model_info(model_info, str(output_dir))
            
            result = {
                'success': True,
                'onnx_path': optimized_path,
                'validation_passed': is_valid,
                'model_info': model_info
            }
            
            self.logger.info("ONNX导出流程完成!")
            return result
            
        except Exception as e:
            self.logger.error(f"ONNX导出流程失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class ONNXInference:
    """ONNX模型推理器"""
    
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """使用ONNX模型进行预测"""
        try:
            # 确保输入数据类型正确
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
                
            # 执行推理
            ort_inputs = {self.input_name: input_data}
            output = self.session.run([self.output_name], ort_inputs)[0]
            
            return output
            
        except Exception as e:
            self.logger.error(f"ONNX推理失败: {e}")
            raise
            
    def predict_batch(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """批量预测"""
        results = []
        
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            batch_output = self.predict(batch)
            results.append(batch_output)
            
        return np.concatenate(results, axis=0)
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'input_name': self.input_name,
            'output_name': self.output_name,
            'input_shape': self.input_shape,
            'providers': self.session.get_providers()
        }


def main():
    """ONNX导出主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX模型导出工具")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='PyTorch模型路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='ONNX模型输出路径'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批量大小'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=96,
        help='序列长度'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=1,
        help='输入特征维度'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=11,
        help='ONNX opset版本'
    )
    
    args = parser.parse_args()
    
    # 创建导出配置
    config = ONNXExportConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        input_shape=(args.batch_size, args.sequence_length, args.input_size),
        opset_version=args.opset_version
    )
    
    # 执行导出
    exporter = ONNXExporter(config)
    result = exporter.export()
    
    if result['success']:
        print("ONNX导出成功!")
        print(f"输出路径: {result['onnx_path']}")
        print(f"验证结果: {'通过' if result['validation_passed'] else '失败'}")
    else:
        print(f"ONNX导出失败: {result['error']}")


if __name__ == "__main__":
    main()
