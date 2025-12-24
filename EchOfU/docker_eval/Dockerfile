# 使用带 CUDA 支持的 PyTorch 官方镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置国内源加速 pip (可选)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 不要弹出任何选择时区的对话框，直接静默安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统级依赖 (OpenCV 必需)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 1. 先拷贝依赖文件并安装 (利用 Docker 缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 拷贝所有代码文件
COPY . .

# 3. 赋予脚本执行权限
RUN chmod +x run_eval.sh

# 4. 设置入口
CMD ["./run_eval.sh"]