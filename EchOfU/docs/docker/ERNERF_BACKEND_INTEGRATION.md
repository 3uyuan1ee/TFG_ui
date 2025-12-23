# ER-NeRF Docker 集成使用指南

## 概述

Backend已经完全集成Docker化的ER-NeRF，支持通过环境变量切换**Docker模式**和**直接调用模式**。

## 文件结构

```
EchOfU/
├── backend/
│   ├── ernerf_docker_client.py      # ER-NeRF Docker客户端
│   ├── video_generator.py            # 已修改，支持Docker调用
│   └── model_trainer.py              # 已修改，支持Docker调用
├── Dockerfile.ernerf                 # ER-NeRF Docker镜像
├── docker-compose.ernerf.yml         # Docker Compose配置
└── docker/ernerf/scripts/
    └── entrypoint.sh                  # 容器启动脚本
```

## 使用方式

### 方式1: 环境变量控制（推荐）

默认使用Docker模式，通过环境变量控制：

```bash
# 使用Docker模式（默认）
export USE_DOCKER_FOR_ERNERF=true

# 使用直接调用模式（需要本地环境）
export USE_DOCKER_FOR_ERNERF=false
```

### 方式2: 代码修改

在 `backend/video_generator.py` 和 `backend/model_trainer.py` 中修改：

```python
USE_DOCKER_FOR_ERNERF = True   # Docker模式
# 或
USE_DOCKER_FOR_ERNERF = False  # 直接调用模式
```

## 工作流程

### 1. 首次准备

```bash
# 1. 进入项目目录
cd /path/to/EchOfU

# 2. 准备BFM模型（见ERNERF_DOCKER.md）
# 将下载的BFM模型放到 ./bfm_models/

# 3. 转换BFM模型
docker compose -f docker-compose.ernerf.yml run --rm ernerf convert-bfm

# 4. 构建Docker镜像
docker compose -f docker-compose.ernerf.yml build

# 5. 启动ER-NeRF容器
docker compose -f docker-compose.ernerf.yml up -d
```

### 2. 启动Backend服务

```bash
# 确保使用Docker模式（默认已启用）
export USE_DOCKER_FOR_ERNERF=true

# 启动Flask服务
python app.py
```

### 3. 通过Web界面使用

访问 `http://localhost:5001`，使用训练和视频生成功能：

#### 训练流程

1. 上传参考视频
2. 选择模型：**ER-NeRF**
3. 选择GPU
4. 点击训练

Backend会自动调用Docker容器：
- 步骤1: 数据预处理（提取图像、音频、landmarks等）
- 步骤2: 三阶段训练（head → lips → torso）

#### 推理流程

1. 选择训练好的ER-NeRF模型
2. 上传参考音频或输入文本（生成语音）
3. 点击生成视频

Backend会自动调用Docker容器：
- 提取DeepSpeech音频特征
- ER-NeRF推理生成视频

## 直接使用Docker客户端

如果需要直接使用ER-NeRF功能（不通过Web界面）：

```python
from backend.ernerf_docker_client import get_ernerf_docker_client

# 获取客户端
client = get_ernerf_docker_client()

# 1. 数据预处理
success, message = client.preprocess(
    video_path="/path/to/video.mp4",
    task_id="my_task"
)

# 2. 模型训练
success, message = client.train(
    data_path="data/my_task",
    model_path="models/ER-NeRF/my_task",
    gpu_id=0,
    stage="auto"  # auto|head|lips|torso
)

# 3. 提取音频特征
success, npy_path = client.extract_audio_features(
    wav_path="/path/to/audio.wav"
)

# 4. 模型推理
success, video_path = client.infer(
    data_path="data/my_task",
    model_path="models/ER-NeRF/my_task",
    audio_npy_path="/path/to/audio.npy",
    with_torso=True
)
```

## 路径映射

### Docker容器内部路径

| 主机路径 | 容器路径 | 说明 |
|----------|----------|------|
| `./data` | `/workspace/data` | 输入视频和预处理数据 |
| `./models/ER-NeRF` | `/workspace/models/ER-NeRF` | 训练模型 |
| `./results` | `/workspace/ER-NeRF/results` | 输出视频 |
| `./bfm_models` | `/workspace/ER-NeRF/data_utils/face_tracking/3DMM` | BFM模型 |

### Backend路径管理

Backend使用`PathManager`管理路径，自动处理相对路径和绝对路径转换。

Docker客户端会自动检测并转换路径格式，确保容器内部可以访问。

## 日志和调试

### 查看Backend日志

```bash
# 如果使用Gunicorn
tail -f logs/access.log
tail -f logs/error.log

# 或直接查看终端输出
```

### 查看Docker日志

```bash
# 查看容器日志
docker compose -f docker-compose.ernerf.yml logs -f ernerf

# 查看最近的日志
docker compose -f docker-compose.ernerf.yml logs --tail=100 ernerf
```

### 进入容器调试

```bash
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell
```

## 常见问题

### Q1: Docker容器未启动

**错误**: `docker: no such service: ernerf`

**解决**:
```bash
docker compose -f docker-compose.ernerf.yml up -d
```

### Q2: 视频文件路径找不到

**错误**: `视频文件不存在: xxx.mp4`

**解决**:
- Docker模式下，确保视频在项目目录下可访问
- 使用绝对路径或将视频放到 `./data/` 目录

### Q3: 训练卡住不动

**可能原因**: 预处理时间较长（可能几小时）

**解决**:
- 查看Docker日志确认进度
- 或进入容器查看进程：`docker compose -f docker-compose.ernerf.yml run --rm ernerf shell`

### Q4: 推理失败 - 找不到音频特征文件

**错误**: `音频特征文件不存在: xxx.npy`

**解决**:
- Docker会自动提取特征到音频文件同目录
- 检查文件权限

## 性能优化

### 1. GPU资源分配

docker-compose中可以限制GPU使用：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']
          capabilities: [gpu]
```

### 2. 共享内存

预处理和训练需要较大共享内存：

```yaml
shm_size: '16gb'  # 在docker-compose.yml中设置
```

### 3. 批量处理

如果需要处理多个任务，考虑：
- 使用多个容器（不同GPU）
- 或使用队列系统（Celery + Redis）

## 切换到直接调用模式

如果不想使用Docker（已配置本地ER-NeRF环境）：

```bash
# 1. 安装ER-NeRF依赖
cd ER-NeRF
conda create -n ernerf python=3.10
conda activate ernerf
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow

# 2. 下载预训练模型（见ERNERF_DOCKER.md）

# 3. 编译CUDA扩展
cd raymarching && python setup.py build_ext --inplace && cd ..
cd gridencoder && python setup.py build_ext --inplace && cd ..
cd shencoder && python setup.py build_ext --inplace && cd ..
cd freqencoder && python setup.py build_ext --inplace && cd ..

# 4. 设置环境变量
export USE_DOCKER_FOR_ERNERF=false

# 5. 启动Backend
python app.py
```

## 对比：Docker模式 vs 直接模式

| 特性 | Docker模式 | 直接模式 |
|------|-----------|----------|
| **环境隔离** | ✅ 完全隔离 | ❌ 共享系统环境 |
| **依赖管理** | ✅ 自动安装 | ❌ 手动配置 |
| **可移植性** | ✅ 跨机器运行 | ❌ 绑定特定环境 |
| **性能** | ⚠️ 轻微开销 | ✅ 原生性能 |
| **调试** | ⚠️ 需要进入容器 | ✅ 直接调试 |
| **部署** | ✅ 生产环境友好 | ⚠️ 需要手动配置 |
| **GPU共享** | ✅ 灵活分配 | ⚠️ 需要手动管理 |

## 总结

推荐使用**Docker模式**，因为：
1. 环境一致性好
2. 部署简单
3. 易于扩展
4. 隔离性强

只有以下情况考虑直接模式：
- 开发调试需要频繁修改ER-NeRF代码
- 性能要求极高（Docker轻微开销不可接受）
- 不想在服务器上安装Docker

## 参考文档

- [ER-NeRF Docker完整指南](ERNERF_DOCKER.md)
- [ER-NeRF官方说明](https://github.com/Fictionarry/ER-NeRF)
