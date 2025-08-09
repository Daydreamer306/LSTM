#!/usr/bin/env python3
"""
PAI-DSW部署状态检查脚本
检查项目在PAI-DSW环境中的部署状态和运行健康度
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

def check_pai_dsw_environment() -> Dict[str, Any]:
    """检查PAI-DSW环境"""
    env_info = {
        'is_pai_dsw': False,
        'workspace_path': None,
        'python_version': sys.version,
        'working_directory': os.getcwd()
    }
    
    if os.path.exists('/mnt/workspace'):
        env_info['is_pai_dsw'] = True
        env_info['workspace_path'] = '/mnt/workspace'
        
    return env_info

def check_gpu_status() -> Dict[str, Any]:
    """检查GPU状态"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_models': [],
        'gpu_memory': []
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_info['gpu_models'].append(gpu_name)
                gpu_info['gpu_memory'].append(f"{gpu_memory:.1f}GB")
                
    except ImportError:
        pass
    
    return gpu_info

def check_dependencies() -> Dict[str, bool]:
    """检查关键依赖包"""
    critical_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'sklearn', 'yaml', 'tqdm'
    ]
    
    dep_status = {}
    for package in critical_packages:
        try:
            __import__(package)
            dep_status[package] = True
        except ImportError:
            dep_status[package] = False
            
    return dep_status

def check_project_structure() -> Dict[str, bool]:
    """检查项目文件结构"""
    required_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/training/__init__.py',
        'src/inference/__init__.py',
        'src/utils/__init__.py',
        'configs/lstm_transformer.yaml',
        'requirements.txt',
        'setup_pai_dsw.py'
    ]
    
    structure_status = {}
    for file_path in required_files:
        structure_status[file_path] = os.path.exists(file_path)
        
    return structure_status

def check_data_files() -> Dict[str, Any]:
    """检查数据文件状态"""
    data_info = {
        'raw_data_found': False,
        'processed_data_found': False,
        'raw_data_files': [],
        'processed_data_files': []
    }
    
    # 检查原始数据
    raw_data_paths = ['data/raw/*.csv', 'data/*.csv']
    for pattern in raw_data_paths:
        import glob
        files = glob.glob(pattern)
        if files:
            data_info['raw_data_found'] = True
            data_info['raw_data_files'].extend(files)
            
    # 检查预处理数据
    processed_paths = ['data/processed/*.npz']
    for pattern in processed_paths:
        files = glob.glob(pattern)
        if files:
            data_info['processed_data_found'] = True
            data_info['processed_data_files'].extend(files)
            
    return data_info

def check_model_files() -> Dict[str, Any]:
    """检查模型文件状态"""
    model_info = {
        'trained_model_exists': False,
        'onnx_model_exists': False,
        'checkpoint_files': [],
        'model_size_mb': 0
    }
    
    # 检查训练好的模型
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        model_info['trained_model_exists'] = True
        model_info['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
        
    # 检查ONNX模型
    onnx_path = 'models/model.onnx'
    if os.path.exists(onnx_path):
        model_info['onnx_model_exists'] = True
        
    # 检查检查点文件
    if os.path.exists('checkpoints'):
        import glob
        checkpoint_files = glob.glob('checkpoints/*.pt')
        model_info['checkpoint_files'] = checkpoint_files
        
    return model_info

def check_training_status() -> Dict[str, Any]:
    """检查训练状态"""
    training_info = {
        'training_history_exists': False,
        'training_completed': False,
        'last_epoch': 0,
        'best_metrics': {}
    }
    
    history_path = 'results/training_history.json'
    if os.path.exists(history_path):
        training_info['training_history_exists'] = True
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            if 'train_loss' in history and len(history['train_loss']) > 0:
                training_info['training_completed'] = True
                training_info['last_epoch'] = len(history['train_loss'])
                
                # 提取最佳指标
                if 'val_mae' in history:
                    training_info['best_metrics']['val_mae'] = min(history['val_mae'])
                if 'val_r2' in history:
                    training_info['best_metrics']['val_r2'] = max(history['val_r2'])
                    
        except Exception as e:
            print(f"解析训练历史失败: {e}")
            
    return training_info

def check_results_and_outputs() -> Dict[str, Any]:
    """检查结果和输出文件"""
    results_info = {
        'evaluation_metrics_exists': False,
        'visualizations_found': False,
        'analysis_reports_found': False,
        'visualization_count': 0
    }
    
    # 检查评估指标
    metrics_path = 'results/evaluation_metrics.json'
    if os.path.exists(metrics_path):
        results_info['evaluation_metrics_exists'] = True
        
    # 检查可视化文件
    if os.path.exists('results'):
        import glob
        viz_files = glob.glob('results/*.png')
        if viz_files:
            results_info['visualizations_found'] = True
            results_info['visualization_count'] = len(viz_files)
            
    # 检查分析报告
    if os.path.exists('analysis'):
        report_files = glob.glob('analysis/*.json') + glob.glob('analysis/*.md')
        if report_files:
            results_info['analysis_reports_found'] = True
            
    return results_info

def check_system_resources() -> Dict[str, Any]:
    """检查系统资源"""
    resource_info = {
        'disk_usage_gb': 0,
        'available_space_gb': 0,
        'memory_usage_gb': 0
    }
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        resource_info['disk_usage_gb'] = used / 1e9
        resource_info['available_space_gb'] = free / 1e9
    except:
        pass
        
    try:
        import psutil
        memory = psutil.virtual_memory()
        resource_info['memory_usage_gb'] = memory.used / 1e9
    except ImportError:
        pass
        
    return resource_info

def run_quick_functionality_test() -> Dict[str, bool]:
    """运行快速功能测试"""
    test_results = {}
    
    # 测试1: 数据加载
    try:
        from src.data.quick_check import quick_data_check
        result = quick_data_check()
        test_results['data_loading'] = result
    except Exception as e:
        test_results['data_loading'] = False
        
    # 测试2: 模型创建
    try:
        from src.models.model_utils import ModelFactory
        factory = ModelFactory()
        config = {'input_size': 1, 'hidden_size': 64, 'sequence_length': 96}
        model = factory.create_model(config, 'basic')
        test_results['model_creation'] = True
    except Exception as e:
        test_results['model_creation'] = False
        
    # 测试3: 配置加载
    try:
        import yaml
        with open('configs/lstm_transformer.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        test_results['config_loading'] = True
    except Exception as e:
        test_results['config_loading'] = False
        
    return test_results

def generate_health_report() -> Dict[str, Any]:
    """生成完整的健康检查报告"""
    print("🔍 开始PAI-DSW部署状态检查...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': check_pai_dsw_environment(),
        'gpu': check_gpu_status(),
        'dependencies': check_dependencies(),
        'project_structure': check_project_structure(),
        'data_files': check_data_files(),
        'model_files': check_model_files(),
        'training_status': check_training_status(),
        'results': check_results_and_outputs(),
        'system_resources': check_system_resources(),
        'functionality_tests': run_quick_functionality_test()
    }
    
    return report

def print_health_summary(report: Dict[str, Any]):
    """打印健康检查摘要"""
    print("\n" + "="*60)
    print("PAI-DSW 部署状态检查报告")
    print("="*60)
    
    # 环境信息
    env = report['environment']
    print(f"\n📋 环境信息:")
    print(f"  PAI-DSW环境: {'✅ 是' if env['is_pai_dsw'] else '❌ 否'}")
    print(f"  工作目录: {env['working_directory']}")
    print(f"  Python版本: {env['python_version'].split()[0]}")
    
    # GPU信息
    gpu = report['gpu']
    print(f"\n🖥️  GPU信息:")
    if gpu['cuda_available']:
        print(f"  CUDA可用: ✅ 是 ({gpu['gpu_count']} 个GPU)")
        for i, (model, memory) in enumerate(zip(gpu['gpu_models'], gpu['gpu_memory'])):
            print(f"    GPU {i}: {model} ({memory})")
    else:
        print(f"  CUDA可用: ❌ 否 (将使用CPU)")
    
    # 依赖状态
    deps = report['dependencies']
    print(f"\n📦 关键依赖:")
    missing_deps = [pkg for pkg, status in deps.items() if not status]
    if missing_deps:
        print(f"  ❌ 缺失依赖: {', '.join(missing_deps)}")
    else:
        print(f"  ✅ 所有关键依赖已安装")
    
    # 项目结构
    struct = report['project_structure']
    print(f"\n📁 项目结构:")
    missing_files = [file for file, status in struct.items() if not status]
    if missing_files:
        print(f"  ❌ 缺失文件: {len(missing_files)} 个")
        for file in missing_files[:3]:  # 只显示前3个
            print(f"    - {file}")
    else:
        print(f"  ✅ 项目结构完整")
    
    # 数据状态
    data = report['data_files']
    print(f"\n💾 数据状态:")
    print(f"  原始数据: {'✅ 已发现' if data['raw_data_found'] else '❌ 未发现'}")
    print(f"  预处理数据: {'✅ 已生成' if data['processed_data_found'] else '❌ 未生成'}")
    
    # 模型状态
    model = report['model_files']
    print(f"\n🤖 模型状态:")
    print(f"  训练模型: {'✅ 存在' if model['trained_model_exists'] else '❌ 不存在'}")
    if model['trained_model_exists']:
        print(f"    大小: {model['model_size_mb']:.1f} MB")
    print(f"  ONNX模型: {'✅ 存在' if model['onnx_model_exists'] else '❌ 不存在'}")
    
    # 训练状态
    training = report['training_status']
    print(f"\n🚀 训练状态:")
    print(f"  训练历史: {'✅ 存在' if training['training_history_exists'] else '❌ 不存在'}")
    print(f"  训练完成: {'✅ 是' if training['training_completed'] else '❌ 否'}")
    if training['training_completed']:
        print(f"    训练轮数: {training['last_epoch']}")
        for metric, value in training['best_metrics'].items():
            print(f"    最佳{metric}: {value:.6f}")
    
    # 结果状态
    results = report['results']
    print(f"\n📊 结果状态:")
    print(f"  评估指标: {'✅ 存在' if results['evaluation_metrics_exists'] else '❌ 不存在'}")
    print(f"  可视化图表: {'✅ 存在' if results['visualizations_found'] else '❌ 不存在'}")
    if results['visualizations_found']:
        print(f"    图表数量: {results['visualization_count']} 个")
    
    # 功能测试
    tests = report['functionality_tests']
    print(f"\n🧪 功能测试:")
    passed_tests = sum(1 for result in tests.values() if result)
    total_tests = len(tests)
    print(f"  测试结果: {passed_tests}/{total_tests} 通过")
    
    for test_name, result in tests.items():
        status = "✅" if result else "❌"
        print(f"    {status} {test_name}")
    
    # 系统资源
    resources = report['system_resources']
    print(f"\n💻 系统资源:")
    if resources['available_space_gb'] > 0:
        print(f"  可用存储: {resources['available_space_gb']:.1f} GB")
        if resources['available_space_gb'] < 5:
            print(f"    ⚠️  存储空间不足")
    
    # 综合评分
    total_checks = 0
    passed_checks = 0
    
    # 环境检查
    total_checks += 1
    if env['is_pai_dsw']:
        passed_checks += 1
        
    # 依赖检查
    total_checks += len(deps)
    passed_checks += sum(1 for status in deps.values() if status)
    
    # 功能测试
    total_checks += len(tests)
    passed_checks += sum(1 for result in tests.values() if result)
    
    # 关键文件检查
    total_checks += 4  # 数据、模型、训练、结果
    if data['raw_data_found']:
        passed_checks += 1
    if model['trained_model_exists']:
        passed_checks += 1
    if training['training_completed']:
        passed_checks += 1
    if results['evaluation_metrics_exists']:
        passed_checks += 1
        
    health_score = (passed_checks / total_checks) * 100
    
    print(f"\n🎯 部署健康度评分: {health_score:.1f}% ({passed_checks}/{total_checks})")
    
    if health_score >= 90:
        print("🎉 部署状态优秀! 所有功能正常运行")
    elif health_score >= 70:
        print("👍 部署状态良好，可以正常使用")
    elif health_score >= 50:
        print("⚠️  部署状态一般，建议检查问题项")
    else:
        print("❌ 部署状态较差，需要解决关键问题")

def save_report(report: Dict[str, Any]):
    """保存检查报告"""
    report_path = 'deployment_health_report.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
    print(f"\n📋 详细报告已保存: {report_path}")

def main():
    """主函数"""
    try:
        # 生成健康检查报告
        report = generate_health_report()
        
        # 打印摘要
        print_health_summary(report)
        
        # 保存详细报告
        save_report(report)
        
        # 提供改进建议
        print("\n💡 改进建议:")
        
        if not report['data_files']['raw_data_found']:
            print("  1. 上传训练数据到 data/raw/ 目录")
            
        if not report['model_files']['trained_model_exists']:
            print("  2. 运行模型训练: python src/training/train.py --config configs/lstm_transformer.yaml")
            
        if not report['results']['evaluation_metrics_exists']:
            print("  3. 运行模型评估: python src/inference/predict.py")
            
        if not all(report['functionality_tests'].values()):
            print("  4. 检查功能测试失败的模块")
            
        print("\n📚 更多帮助:")
        print("  - 查看部署指南: PAI_DSW_DEPLOYMENT.md")
        print("  - 运行快速启动: PAI_DSW_QuickStart.ipynb")
        print("  - 一键部署脚本: bash deploy_pai_dsw_full.sh")
        
    except Exception as e:
        print(f"❌ 健康检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
