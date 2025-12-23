# ER-NeRF Docker 完整部署指南

## 目录

1. [系统要求](#系统要求)
2. [快速开始](#快速开始)
3. [详细部署步骤](#详细部署步骤)
4. [Backend集成](#backend集成)
5. [故障排查](#故障排查)
6. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **GPU** | NVIDIA RTX 3090 (24GB) | NVIDIA RTX 4090 (24GB) 或 A100 (40GB+) |
| **CPU** | 8核心 | 16核心+ |
| **内存** | 32GB | 64GB+ |
| **存储** | 100GB SSD | 500GB NVMe SSD |

### 软件要求

| 软件 | 版本要求 |
|------|----------|
| **操作系统** | Ubuntu 18.04 / 20.04 / 22.04 |
| **NVIDIA Driver** | ≥ 470.x |
| **Docker** | ≥ 20.10 |
| **NVIDIA Container Toolkit** | ≥ 1.6.0 |
| **Python** | ≥ 3.8 (仅在开发环境需要) |

### 不支持的系统

- ❌ **macOS** (不支持CUDA)
- ❌ **Windows** (支持有限，建议WSL2)
- ✅ **Linux** (完整支持)

---

## 快速开始

### 5分钟快速部署

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd EchOfU

# 2. 运行自动部署脚本
bash scripts/start_ernerf_docker.sh setup

# 3. 等待构建完成（30-60分钟）

# 4. 验证安装
docker compose -f docker-compose.ernerf.yml run --rm ernerf python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 5. 测试预处理（需要准备视频）
docker compose -f docker-compose.ernerf.yml run --rm ernerf preprocess /path/to/test.mp4 test_task
```

---

## 详细部署步骤

### 步骤1: 准备服务器环境

```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu18.04 nvidia-smi
```

### 步骤2: 下载项目代码

```bash
# 克隆项目（使用你的实际仓库地址）
git clone <your-repo-url>
cd EchOfU

# 检查项目结构
ls -la
```

应该看到：
```
EchOfU/
├── Dockerfile.ernerf
├── docker-compose.ernerf.yml
├── backend/
├── ER-NeRF/
├── docker/
└── scripts/
```

### 步骤3: 准备BFM模型

**重要**: BFM模型需要手动申请下载。

#### 3.1 申请BFM模型

1. 访问: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
2. 使用学术邮箱注册账号
3. 申请下载权限（通常1-3个工作日）
4. 下载BFM模型文件

#### 3.2 放置模型文件

```bash
# 创建模型目录
mkdir -p ./bfm_models

# 将下载的文件放到该目录
# 文件名通常是: BFM_model_front.mat 或类似名称
```

#### 3.3 转换BFM模型

```bash
# 使用Docker转换
docker compose -f docker-compose.ernerf.yml run --rm ernerf convert-bfm

# 验证转换结果
ls ./bfm_models/
# 应该看到:
# exp_para.npy
# tex_para.npy
# shape_para.npy
```

### 步骤4: 构建Docker镜像

```bash
# 方式1: 使用脚本（推荐）
bash scripts/start_ernerf_docker.sh build

# 方式2: 直接使用docker compose
docker compose -f docker-compose.ernerf.yml build
```

**预计时间**: 30-60分钟

**构建过程**:
1. 拉取基础镜像 (5分钟)
2. 安装系统依赖 (5分钟)
3. 安装Miniconda (2分钟)
4. 安装PyTorch (5分钟)
5. 安装Python包 (10分钟)
6. 编译pytorch3d (15分钟)
7. 编译CUDA扩展 (10分钟)
8. 下载预训练模型 (5分钟)

### 步骤5: 启动服务

```bash
# 启动ER-NeRF容器
docker compose -f docker-compose.ernerf.yml up -d

# 查看容器状态
docker compose -f docker-compose.ernerf.yml ps

# 查看日志
docker compose -f docker-compose.ernerf.yml logs -f
```

### 步骤6: 验证安装

```bash
# 运行测试脚本
docker compose -f docker-compose.ernerf.yml run --rm ernerf python -c "
import torch
import torch3d
print('=' * 50)
print('ER-NeRF 环境验证')
print('=' * 50)
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print('pytorch3d导入: 成功')
print('=' * 50)
print('所有检查通过！')
"
```

---

## Backend集成

### 启动Backend服务

```bash
# 1. 设置环境变量（默认已启用Docker）
export USE_DOCKER_FOR_ERNERF=true

# 2. 安装Backend依赖
pip install -r requirements.txt

# 3. 启动Flask服务
python app.py
```

### 测试完整流程

#### 1. 准备测试视频

```bash
# 将测试视频放到项目目录
cp /path/to/test.mp4 ./data/test_video.mp4
```

#### 2. 通过API训练

```bash
# 方式1: 使用curl
curl -X POST http://localhost:5001/api/train \
  -F "ref_video=@./data/test_video.mp4" \
  -F "model_choice=ER-NeRF" \
  -F "gpu_choice=GPU0" \
  -F "speaker_id=test_task"
```

#### 3. 通过API推理

```bash
curl -X POST http://localhost:5001/api/generate_video \
  -F "ref_audio=@./data/test.wav" \
  -F "model_name=ER-NeRF" \
  -F "model_param=models/ER-NeRF/test_task" \
  -F "gpu_choice=GPU0"
```

### 监控日志

```bash
# Backend日志
tail -f logs/access.log
tail -f logs/error.log

# Docker日志
docker compose -f docker-compose.ernerf.yml logs -f ernerf
```

---

## 故障排查

### 问题1: Docker构建失败

**症状**: 构建过程中断

**原因**: 网络问题、依赖冲突

**解决方案**:

```bash
# 清理Docker缓存
docker system prune -a

# 重新构建
docker compose -f docker-compose.ernerf.yml build --no-cache
```

### 问题2: CUDA扩展编译失败

**症状**: `error: CUDA toolkit not found`

**解决方案**:

```bash
# 检查nvidia-container-toolkit
which nvidia-container-runtime

# 重新安装
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 问题3: GPU内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:

1. 减小batch size
2. 使用多GPU
3. 降低图像分辨率

```bash
# 修改docker-compose.yml
services:
  ernerf:
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 问题4: 预处理卡住

**症状**: preprocess命令无响应

**原因**: 预处理需要很长时间（几小时）

**解决方案**:

```bash
# 查看进程状态
docker compose -f docker-compose.ernerf.yml top ernerf

# 进入容器查看
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell
cd /workspace/ER-NeRF
ps aux | grep python
```

### 问题5: 容器无法访问主机文件

**症状**: `No such file or directory`

**解决方案**:

```bash
# 检查卷挂载
docker compose -f docker-compose.ernerf.yml config

# 确保路径存在
ls -la ./data
ls -la ./models/ER-NeRF

# 修改权限
chmod -R 777 ./data ./models ./results
```

---

## 常见问题

### Q1: 可以在Mac上运行吗？

**A**: 不能。ER-NeRF需要NVIDIA GPU + CUDA，Mac不支持。

**解决方案**:
- 使用远程Linux服务器
- 使用云GPU平台（AutoDL、AWS等）
- 参考文档中"Mac用户指南"

### Q2: 训练需要多长时间？

**A**: 取决于视频长度和硬件配置

| 阶段 | RTX 3090 | RTX 4090 |
|------|----------|----------|
| 预处理 | 2-4小时 | 1-2小时 |
| 头部训练 | 2-3小时 | 1-2小时 |
| 嘴唇微调 | 1-2小时 | 0.5-1小时 |
| 躯干训练 | 3-4小时 | 2-3小时 |
| **总计** | **8-13小时** | **5-8小时** |

### Q3: 需要多少GPU内存？

**A**: 最少24GB，推荐40GB+

| GPU内存 | 分辨率 | 训练 | 推理 |
|---------|--------|------|------|
| 12GB | 512x512 | ❌ | ✅ |
| 16GB | 512x512 | ⚠️ | ✅ |
| 24GB | 512x512 | ✅ | ✅ |
| 40GB+ | 1024x1024 | ✅ | ✅ |

### Q4: 可以多GPU并行训练吗？

**A**: 可以

```bash
# 启动第二个GPU服务
docker compose -f docker-compose.ernerf.yml --profile gpu1 up -d

# 使用GPU1训练
docker compose -f docker-compose.ernerf.yml run --rm ernerf-gpu1 train data/task models/ER-NeRF/task
```

### Q5: 如何备份模型和数据？

**A**: 定期备份重要目录

```bash
# 创建备份脚本
cat > backup.sh <<'EOF'
#!/bin/bash
BACKUP_DIR="/backup/ernerf_$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份模型
cp -r ./models/ER-NeRF $BACKUP_DIR/

# 备份BFM模型
cp -r ./bfm_models $BACKUP_DIR/

# 压缩
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "备份完成: $BACKUP_DIR.tar.gz"
EOF

chmod +x backup.sh

# 定期执行（例如每天凌晨3点）
crontab -e
# 添加: 0 3 * * * /path/to/EchOfU/backup.sh
```

---

## 性能优化

### 1. 使用NVMe SSD

```bash
# 检查磁盘类型
lsblk -d -o name,rota

# rota=0 表示SSD
# rota=1 表示HDD（需要更换）
```

### 2. 优化共享内存

```yaml
# docker-compose.ernerf.yml
services:
  ernerf:
    shm_size: '32gb'  # 增加到32GB
```

### 3. 使用多GPU

```bash
# 修改数据预处理脚本，支持多GPU
# 见ER-NeRF文档
```

---

## 维护

### 更新镜像

```bash
# 拉取最新代码
git pull

# 重新构建
docker compose -f docker-compose.ernerf.yml build --no-cache

# 重启服务
docker compose -f docker-compose.ernerf.yml up -d
```

### 清理磁盘空间

```bash
# 清理未使用的Docker资源
docker system prune -a --volumes

# 清理构建缓存
docker builder prune -a
```

### 监控系统资源

```bash
# GPU使用率
watch -n 1 nvidia-smi

# 磁盘使用
df -h

# 内存使用
free -h

# Docker容器资源
docker stats
```

---

## 安全建议

1. **不要在公网直接暴露容器**
   ```bash
   # 仅监听本地
   ports:
     - "127.0.0.1:5001:5001"
   ```

2. **使用环境变量管理敏感信息**
   ```bash
   # .env
   CUDA_VISIBLE_DEVICES=0
   API_KEY=your_secret_key
   ```

3. **定期更新Docker镜像**
   ```bash
   docker compose -f docker-compose.ernerf.yml pull
   docker compose -f docker-compose.ernerf.yml up -d
   ```

---

## 参考文档

- [ER-NeRF Docker完整指南](ERNERF_DOCKER.md)
- [Backend集成指南](ERNERF_BACKEND_INTEGRATION.md)
- [ER-NeRF官方论文](https://arxiv.org/abs/2212.08028)
- [Docker官方文档](https://docs.docker.com/)
- [NVIDIA Docker文档](https://github.com/NVIDIA/nvidia-docker)

---

## 获取帮助

如遇到问题：
1. 查看日志: `docker compose logs -f ernerf`
2. 检查故障排查章节
3. 提交Issue到项目仓库
4. 联系项目维护者
