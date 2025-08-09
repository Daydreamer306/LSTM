# PAI-DSW Jupyter Notebook 环境配置
# 在PAI-DSW Jupyter环境中运行此代码进行环境初始化

import os
import sys
import subprocess
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置PAI-DSW工作环境
if os.path.exists('/mnt/workspace'):
    os.chdir('/mnt/workspace')
    sys.path.insert(0, '/mnt/workspace')
else:
    print("警告: 未检测到PAI-DSW环境，使用当前目录")

print("=== PAI-DSW LSTM+Transformer 项目初始化 ===")

# 检查环境
def check_environment():
    """检查PAI-DSW环境配置"""
    print("1. 检查Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    print("\n2. 检查深度学习环境...")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch未安装，将在安装依赖时自动安装")
    
    print("\n3. 检查存储空间...")
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        print(f"总空间: {total / 1e9:.1f} GB")
        print(f"已用空间: {used / 1e9:.1f} GB") 
        print(f"可用空间: {free / 1e9:.1f} GB")
        
        if free / 1e9 < 10:
            print("⚠️  警告: 可用存储空间不足10GB")
    except:
        print("无法获取存储信息")
    
    print("\n4. 检查网络连接...")
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=5)
        print("网络连接正常 ✓")
    except:
        print("网络连接异常，可能影响依赖安装")

# 安装依赖
def install_dependencies():
    """安装项目依赖"""
    print("\n5. 安装项目依赖...")
    
    # 检查requirements.txt是否存在
    if not os.path.exists('requirements.txt'):
        print("requirements.txt不存在，创建默认依赖文件...")
        create_requirements_file()
    
    # 使用国内镜像源安装
    mirrors = [
        'https://pypi.tuna.tsinghua.edu.cn/simple/',
        'https://mirrors.aliyun.com/pypi/simple/',
        'https://pypi.douban.com/simple/'
    ]
    
    for mirror in mirrors:
        try:
            print(f"尝试使用镜像源: {mirror}")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt',
                '-i', mirror, '--timeout', '60'
            ], check=True, timeout=300)
            print("✓ 依赖安装完成!")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"镜像源 {mirror} 安装失败: {e}")
            if mirror == mirrors[-1]:
                print("所有镜像源都失败，请手动安装依赖")
                return False
    
    # 验证关键依赖
    print("\n6. 验证关键依赖...")
    critical_packages = ['torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'PyYAML']
    
    missing_packages = []
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"警告: 以下关键依赖缺失: {missing_packages}")
        return False
    
    return True

def create_requirements_file():
    """创建默认的requirements.txt文件"""
    requirements = """torch>=1.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
PyYAML>=6.0
tqdm>=4.62.0
tensorboard>=2.7.0
onnx>=1.12.0
onnxruntime-gpu>=1.12.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("已创建默认requirements.txt文件")

# 创建目录结构
def create_directory_structure():
    """创建项目目录结构"""
    print("\n7. 创建目录结构...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models',
        'results/training',
        'results/predictions',
        'results/visualizations',
        'logs',
        'checkpoints',
        'analysis',
        'exports'
    ]
    
    created_dirs = []
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"✓ {directory}")
        except Exception as e:
            print(f"✗ {directory} - 创建失败: {e}")
    
    print(f"成功创建 {len(created_dirs)} 个目录")
    return created_dirs

def verify_project_structure():
    """验证项目结构完整性"""
    print("\n8. 验证项目结构...")
    
    required_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py', 
        'src/training/__init__.py',
        'src/inference/__init__.py',
        'src/utils/__init__.py',
        'configs/lstm_transformer.yaml',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 缺失")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  警告: 以下关键文件缺失: {missing_files}")
        return False
    
    print("✓ 项目结构完整")
    return True

def create_pai_dsw_config():
    """创建PAI-DSW特定配置"""
    print("\n9. 创建PAI-DSW配置...")
    
    pai_config = {
        "workspace_path": "/mnt/workspace" if os.path.exists('/mnt/workspace') else os.getcwd(),
        "python_path": sys.executable,
        "gpu_available": False,
        "gpu_count": 0,
        "memory_gb": 0,
        "storage_gb": 0
    }
    
    # 检测GPU
    try:
        import torch
        if torch.cuda.is_available():
            pai_config["gpu_available"] = True
            pai_config["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    
    # 检测内存
    try:
        import psutil
        pai_config["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    
    # 检测存储
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        pai_config["storage_gb"] = round(free / 1e9, 1)
    except:
        pass
    
    # 保存配置
    config_path = 'pai_dsw_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(pai_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ PAI-DSW配置已保存: {config_path}")
    return pai_config

def run_quick_test():
    """运行快速功能测试"""
    print("\n10. 运行快速功能测试...")
    
    tests = []
    
    # 测试1: 数据处理
    try:
        from src.data.quick_check import quick_data_check
        result = quick_data_check()
        tests.append(("数据处理模块", result))
        print("✓ 数据处理模块")
    except Exception as e:
        tests.append(("数据处理模块", False))
        print(f"✗ 数据处理模块: {e}")
    
    # 测试2: 模型创建  
    try:
        from src.models.model_utils import ModelFactory
        factory = ModelFactory()
        config = {'input_size': 1, 'hidden_size': 64, 'sequence_length': 96}
        model = factory.create_model(config, 'basic')
        tests.append(("模型创建", True))
        print("✓ 模型创建")
    except Exception as e:
        tests.append(("模型创建", False))
        print(f"✗ 模型创建: {e}")
    
    # 测试3: 配置加载
    try:
        import yaml
        with open('configs/lstm_transformer.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        tests.append(("配置加载", True))
        print("✓ 配置加载")
    except Exception as e:
        tests.append(("配置加载", False))
        print(f"✗ 配置加载: {e}")
    
    # 输出测试结果
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有测试通过，环境配置成功!")
        return True
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return False

def main():
    """主初始化流程"""
    try:
        print("开始PAI-DSW环境初始化...\n")
        
        # 1. 环境检查
        check_environment()
        
        # 2. 安装依赖  
        deps_ok = install_dependencies()
        if not deps_ok:
            print("依赖安装失败，请手动解决后重新运行")
            return False
        
        # 3. 创建目录
        create_directory_structure()
        
        # 4. 验证项目结构
        structure_ok = verify_project_structure()
        if not structure_ok:
            print("项目结构不完整，请检查文件")
            return False
        
        # 5. 创建配置
        pai_config = create_pai_dsw_config()
        
        # 6. 运行测试
        test_ok = run_quick_test()
        
        # 7. 输出摘要
        print("\n" + "="*60)
        print("PAI-DSW 初始化完成摘要")
        print("="*60)
        print(f"工作目录: {os.getcwd()}")
        print(f"Python版本: {sys.version.split()[0]}")
        
        if pai_config.get('gpu_available'):
            print(f"GPU: {pai_config['gpu_count']} 个可用")
        else:
            print("GPU: 不可用 (将使用CPU)")
            
        print(f"内存: {pai_config.get('memory_gb', 'unknown')} GB")
        print(f"存储: {pai_config.get('storage_gb', 'unknown')} GB")
        
        if test_ok:
            print("\n🎉 初始化成功! 可以开始模型训练")
            print("\n推荐下一步操作:")
            print("1. 上传训练数据到 data/raw/ 目录")
            print("2. 运行数据预处理: python src/data/preprocess.py")  
            print("3. 开始模型训练: python src/training/train.py --config configs/lstm_transformer.yaml")
        else:
            print("\n⚠️  初始化完成但存在问题，请检查上述错误信息")
            
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
