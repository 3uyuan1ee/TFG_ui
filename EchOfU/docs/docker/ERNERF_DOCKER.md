# ER-NeRF Docker 封装说明

## 项目概述

ER-NeRF (Audio-Driven Talking Head Generation with Efficient Region-Aware Neural Radiance Fields) 是一个基于音频驱动的虚拟数字人视频生成系统。

### 官方环境要求

- **操作系统**: Ubuntu 18.04
- **CUDA**: 11.3
- **PyTorch**: 1.12.1
- **Python**: 3.10

## 快速开始

### 1. 准备工作

#### 1.1 确保已安装Docker和NVIDIA Container Toolkit

```bash
# 检查Docker
docker --version

# 检查NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu18.04 nvidia-smi
```

#### 1.2 准备BFM模型（重要！）

ER-NeRF需要Basel Face Model (BFM)，该模型需要手动申请下载：

1. 访问 [BFM官网](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details)
2. 注册账号并申请下载
3. 下载后将文件放置到 `./bfm_models/` 目录
4. 运行转换命令：

```bash
docker compose -f docker-compose.ernerf.yml run --rm ernerf convert-bfm
```

### 2. 构建镜像

```bash
# 进入项目目录
cd /path/to/EchOfU

# 构建镜像（首次使用或代码更新时）
docker compose -f docker-compose.ernerf.yml build
```

**预计构建时间**: 约30-60分钟（取决于网络速度）
- 下载基础镜像: ~5分钟
- 安装依赖: ~10分钟
- 编译pytorch3d: ~15分钟
- 编译CUDA扩展: ~10分钟
- 下载预训练模型: ~5分钟

### 3. 启动服务

```bash
# 启动ER-NeRF容器
docker compose -f docker-compose.ernerf.yml up -d

# 查看日志
docker compose -f docker-compose.ernerf.yml logs -f
```

## 完整工作流程

### 流程概览

```
视频输入 → 预处理 → 训练 → 推理 → 输出视频
```

### 步骤1: 数据预处理

将参考视频进行预处理，提取图像、音频特征、landmarks等。

```bash
# 完整预处理命令
docker compose -f docker-compose.ernerf.yml run --rm ernerf preprocess /path/to/video.mp4 my_task
```

**参数说明**:
- `/path/to/video.mp4`: 输入视频路径
- `my_task`: 任务ID（可选，默认使用视频文件名）

**视频要求**:
- 帧率: 25 FPS
- 所有帧必须包含说话人
- 分辨率: ~512x512
- 时长: 1-5分钟

**预处理包含以下步骤**（可能需要几小时）:
1. 提取音频 (ffmpeg)
2. 提取音频特征 (DeepSpeech)
3. 提取图像帧 (25fps)
4. Face parsing
5. 提取背景图
6. 提取躯干和GT图像
7. 提取人脸landmarks (face_alignment)
8. 生成眨眼数据 (EAR算法，已优化)
9. Face tracking (相机参数估计)
10. 生成transforms.json

**输出结果**:
```
./data/my_task/
├── my_task.mp4          # 原始视频
├── aud.wav              # 提取的音频
├── aud.npy              # DeepSpeech特征
├── ori_imgs/            # 原始图像帧
├── parsing/             # Face parsing结果
├── gt_imgs/             # Ground truth图像
├── torso_imgs/          # 躯干图像
├── mask/                # Mask图像
├── landmarks/           # 人脸landmarks
├── au.csv               # 眨眼数据
├── bc.jpg               # 背景图
├── track_params.pt      # 相机参数
├── transforms_train.json # 训练集变换
└── transforms_val.json   # 验证集变换
```

### 步骤2: 模型训练

ER-NeRF采用三阶段训练策略：

```bash
# 自动模式（推荐）- 自动完成所有阶段
docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/my_task models/ER-NeRF/my_task

# 手动指定阶段
docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/my_task models/ER-NeRF/my_task 0 head    # 阶段1
docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/my_task models/ER-NeRF/my_task 0 lips    # 阶段2
docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/my_task models/ER-NeRF/my_task 0 torso   # 阶段3
```

**训练阶段说明**:

| 阶段 | 名称 | 迭代次数 | 用途 | 时间估计 |
|------|------|----------|------|----------|
| 1 | head | 100k | 头部基础训练 + LPIPS | ~2-3小时 |
| 2 | lips | 125k | 嘴唇微调 | ~1-2小时 |
| 3 | torso | 200k | 躯干训练 | ~3-4小时 |

**输出结果**:
```
./models/ER-NeRF/my_task/
├── checkpoints/          # 模型检查点
│   ├── ngp_ep000010.pth
│   ├── ngp_ep000020.pth
│   └── ...
├── results/              # 测试结果
└── opt.txt               # 训练配置
```

### 步骤3: 音频特征提取（用于推理）

如果要用自定义音频驱动虚拟人，需要先提取音频特征：

```bash
docker compose -f docker-compose.ernerf.yml run --rm ernerf extract-features /path/to/audio.wav
```

**输出**: `/path/to/audio.npy`

### 步骤4: 模型推理

使用训练好的模型和音频特征生成视频：

```bash
# 基础推理
docker compose -f docker-compose.ernerf.yml run --rm ernerf test data/my_task models/ER-NeRF/my_task audio.npy

# 不包含躯干（仅头部）
docker compose -f docker-compose.ernerf.yml run --rm ernerf test data/my_task models/ER-NeRF/my_task audio.npy false
```

**输出视频位置**:
- `./models/ER-NeRF/my_task/results/`
- 或 `./results/`

## 高级用法

### 查看容器环境信息

```bash
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell
```

### 多GPU训练

如果有多张GPU，可以启动第二个服务：

```bash
# 启动GPU1服务
docker compose -f docker-compose.ernerf.yml --profile gpu1 up -d

# 使用GPU1训练
docker compose -f docker-compose.ernerf.yml run --rm ernerf-gpu1 train data/task models/ER-NeRF/task
```

### 持久化存储

所有重要数据都通过Docker卷持久化到主机：

| 主机路径 | 容器路径 | 用途 |
|----------|----------|------|
| `./data` | `/workspace/data` | 输入视频和预处理数据 |
| `./models/ER-NeRF` | `/workspace/models/ER-NeRF` | 训练模型 |
| `./results` | `/workspace/ER-NeRF/results` | 输出视频 |
| `./logs` | `/workspace/logs` | 日志文件 |
| `./bfm_models` | `/workspace/ER-NeRF/data_utils/face_tracking/3DMM` | BFM模型 |

### 直接使用Python命令

进入容器后可以直接运行Python脚本：

```bash
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell

# 在容器内
cd /workspace/ER-NeRF

# 预处理
python data_utils/process.py data/obama/obama.mp4 --task -1 --asr_model deepspeech

# 训练
python main.py data/obama/ --workspace models/ER-NeRF/obama -O --iters 100000

# 推理
python main.py data/obama/ --workspace models/ER-NeRF/obama -O --test --aud audio.npy
```

## 常见问题

### Q1: 预处理时找不到BFM模型文件

**错误信息**:
```
[WARN] BFM模型文件未找到！
```

**解决方案**:
1. 访问 https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details
2. 注册并下载BFM模型
3. 将文件放到 `./bfm_models/` 目录
4. 运行: `docker compose -f docker-compose.ernerf.yml run --rm ernerf convert-bfm`

### Q2: face_tracker.py不生成.pt文件

**说明**: 此问题已在 `process.py` 中修复，现在使用直接函数调用而非 `os.system`。

### Q3: au.csv文件缺失

**说明**: 已改用EAR算法自动生成，无需OpenFace。

### Q4: 训练速度慢

**优化措施**:
- 使用 `--fp16` 开启混合精度训练
- 使用 `--cuda_ray` 开启CUDA加速
- face_tracker参数已优化（加快预处理速度）

### Q5: CUDA扩展编译失败

**解决方案**:
- 确保使用NVIDIA Container Toolkit
- 检查GPU驱动版本是否兼容CUDA 11.3

### Q6: pytorch3d安装失败

**解决方案**:
- Dockerfile中使用预编译wheels加速安装
- 如仍失败，检查网络连接

## 系统架构

```
ER-NeRF Docker容器
├── Python 3.10
├── PyTorch 1.12.1 + CUDA 11.3
├── pytorch3d (3D处理)
├── TensorFlow (face parsing)
├── CUDA扩展
│   ├── raymarching
│   ├── gridencoder
│   ├── shencoder
│   └── freqencoder
└── 预训练模型
    ├── Face parsing (79999_iter.pth)
    ├── 3DMM (exp_info.npy等)
    └── BFM (手动下载)
```

## 与主系统集成

ER-NeRF容器可以通过以下方式与EchOfU主系统集成：

### 1. 共享数据目录

```yaml
volumes:
  - ./static/videos/ref_videos:/workspace/data:rw
  - ./static/videos/res_videos:/workspace/ER-NeRF/results:rw
  - ./models/ER-NeRF:/workspace/models/ER-NeRF:rw
```

### 2. 通过Backend调用

在 `backend/model_trainer.py` 和 `backend/video_generator.py` 中，可以通过 `subprocess` 调用Docker命令：

```python
# 训练
subprocess.run([
    "docker", "compose", "-f", "docker-compose.ernerf.yml",
    "run", "--rm", "ernerf", "train",
    f"data/{task_id}", f"models/ER-NeRF/{task_id}"
])

# 推理
subprocess.run([
    "docker", "compose", "-f", "docker-compose.ernerf.yml",
    "run", "--rm", "ernerf", "test",
    f"data/{task_id}", f"models/ER-NeRF/{task_id}", audio_npy
])
```

### 3. 环境变量配置

```bash
# .env
ERNERF_DOCKER_COMPOSE=docker-compose.ernerf.yml
ERNERF_DATA_PATH=./data
ERNERF_MODEL_PATH=./models/ER-NeRF
ERNERF_RESULT_PATH=./results
```

## 参考资料

- [ER-NeRF官方论文](https://arxiv.org/abs/2212.08028)
- [ER-NeRF GitHub](https://github.com/Fictionarry/ER-NeRF)
- [AD-NeRF GitHub](https://github.com/YudongGuo/AD-NeRF)
- [PyTorch3D文档](https://pytorch3d.org/docs/)


