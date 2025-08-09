# PAI-DSW兼容的Docker镜像
# 使用阿里云官方PyTorch镜像，与PAI-DSW环境保持一致
FROM registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch:1.13.1-gpu-py38-cu117-ubuntu20.04

# 设置PAI-DSW工作目录
WORKDIR /mnt/workspace

# 设置环境变量
ENV PYTHONPATH=/mnt/workspace
ENV CUDA_VISIBLE_DEVICES=0
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 使用清华大学镜像源，提高下载速度
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目文件
COPY . .

# 创建PAI-DSW标准目录结构
RUN mkdir -p data models results checkpoints logs

# 设置权限
RUN chmod +x /mnt/workspace && \
    chmod +x setup_pai_dsw.py && \
    chmod +x deploy_pai_dsw.sh

# PAI-DSW环境检查和初始化
RUN python setup_pai_dsw.py

# 默认命令 - 适配PAI-DSW
CMD ["python", "src/training/train.py", "--config", "configs/lstm_transformer.yaml"]
