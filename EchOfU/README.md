## 界面预览

<img src="resource/index.png" alt="首页"/>

## 功能

- **语音克隆**: CosyVoice
- **视频生成**: ER-NeRF
- **模型训练**: 深度学习模型三阶段训练
- **对话系统**: 基于llm的人机交互

## 快速开始

```bash
# 安装
git clone https://github.com/3uyuan1ee/TFG_ui.git
cd TFG_ui
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
cd EchOfU
pip install -r requirements.txt

# 启动
python app.py
```

访问 http://localhost:5001

## 文档

- [完整配置文档](docs/配置文档.md) - CosyVoice、ER-NeRF部署
- [ER-NeRF部署指南](docs/docker/ERNERF_DEPLOYMENT.md)
- [Docker使用说明](docs/docker/ERNERF_DOCKER.md)

## 常见问题

**Q: 子模块初始化失败**
```bash
git submodule update --init --recursive --depth=1
```

**Q: 国内加速**
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 技术栈

Flask 3.0 + PyTorch 2.3 + CosyVoice + ER-NeRF

## 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "前端层 Frontend Layer"
        A1[主页面<br/>index.html]
        A2[视频生成页面<br/>video_generation.html]
        A3[模型训练页面<br/>model_training.html]
        A4[音频克隆页面<br/>audio_clone.html]
        A5[对话系统页面<br/>chat_system.html]
    end

    subgraph "Web服务层 Web Service Layer"
        B1[Flask Web Server<br/>app.py]
        B2[API处理器<br/>api_handlers.py]
        B3[路由控制器<br/>Routes]
        B4[请求验证<br/>Request Validator]
    end

    subgraph "任务调度层 Task Scheduling Layer"
        C1[任务管理器<br/>Task Manager]
        C2[异步任务队列<br/>Async Queue]
        C3[进度监控器<br/>Progress Monitor]
        C4[资源管理器<br/>Resource Manager]
    end

    subgraph "核心服务层 Core Service Layer"
        subgraph "语音服务 Voice Services"
            D1[语音生成器<br/>voice_generator.py]
            D2[音频预处理<br/>audio_preprocessor.py]
            D3[升降调服务<br/>pitch_shift.py]
            D4[音频克隆<br/>CV_clone.py]
        end

        subgraph "视频服务 Video Services"
            D5[视频生成器<br/>video_generator.py]
            D6[模型训练器<br/>model_trainer.py]
            D7[Docker客户端<br/>ernerf_docker_client.py]
        end

        subgraph "对话服务 Chat Services"
            D8[对话引擎<br/>chat_engine.py]
            D9[上下文管理<br/>Context Manager]
        end

        subgraph "数据服务 Data Services"
            D10[文件管理器<br/>file_manager.py]
            D11[路径管理器<br/>path_manager.py]
            D12[模型下载管理<br/>model_download_manager.py]
        end
    end

    subgraph "AI模型层 AI Model Layer"
        E1[CosyVoice<br/>语音合成模型]
        E2[OpenVoice<br/>语音克隆模型]
        E3[ER-NeRF<br/>视频生成模型]
        E4[LLM<br/>大语言模型]
    end

    subgraph "数据存储层 Data Storage Layer"
        F1[模型文件<br/>models/]
        F2[训练数据集<br/>datasets/]
        F3[生成结果<br/>static/videos/]
        F4[音频文件<br/>static/audios/]
        F5[用户配置<br/>configs/]
    end

    subgraph "基础设施层 Infrastructure Layer"
        G1[GPU计算<br/>CUDA]
        G2[Docker容器<br/>Container]
        G3[文件系统<br/>File System]
        G4[系统监控<br/>GPUtil/psutil]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B1

    B1 --> B2
    B2 --> B3
    B2 --> B4

    B3 --> C1
    C1 --> C2
    C1 --> C3
    C1 --> C4

    C2 --> D1
    C2 --> D2
    C2 --> D3
    C2 --> D4
    C2 --> D5
    C2 --> D6
    C2 --> D7
    C2 --> D8
    C2 --> D10
    C2 --> D11
    C2 --> D12

    D1 --> E1
    D4 --> E2
    D5 --> E3
    D6 --> E3
    D7 --> E3
    D8 --> E4

    E1 --> F1
    E2 --> F1
    E3 --> F1
    E4 --> F1

    D5 --> F3
    D1 --> F4
    D6 --> F2
    D10 --> F3
    D10 --> F4

    D1 --> G1
    D5 --> G1
    D6 --> G1
    D7 --> G2
    C4 --> G4
```

### 核心数据流

#### 视频生成流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as 前端页面
    participant W as Flask服务
    participant V as 视频生成器
    participant T as 语音合成
    participant N as ER-NeRF
    participant S as 存储

    U->>F: 提交生成请求
    F->>W: POST /api/generate-video
    W->>V: 调用生成服务
    V->>T: 文本转语音
    T->>T: CosyVoice合成
    T-->>V: 返回音频
    V->>V: 音频变调处理
    V->>N: 调用ER-NeRF
    N->>N: 生成视频帧
    N-->>V: 返回视频路径
    V->>S: 保存视频
    S-->>W: 返回文件路径
    W-->>F: 返回生成结果
    F->>U: 展示视频
```

#### 模型训练流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as 前端页面
    participant W as Flask服务
    participant M as 模型训练器
    participant D as Docker容器
    participant S as 存储

    U->>F: 上传训练数据
    F->>W: POST /api/train-model
    W->>M: 创建训练任务
    M->>M: 数据预处理
    M->>M: 准备配置文件
    M->>D: 启动训练容器
    D->>D: 执行训练脚本
    D->>D: 阶段1训练
    D->>D: 阶段2训练
    D->>D: 阶段3训练
    D->>S: 保存模型检查点
    D-->>M: 返回训练状态
    M->>S: 保存最终模型
    M-->>W: 返回训练结果
    W-->>F: 返回模型路径
    F->>U: 显示训练完成
```

#### 人机对话流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as 前端页面
    participant C as 对话引擎
    participant L as LLM
    participant T as 语音合成
    participant V as 视频生成

    U->>F: 发送语音/文本
    F->>C: POST /api/chat
    C->>L: 发送对话历史
    L->>L: 生成回复
    L-->>C: 返回文本
    C->>T: 文本转语音
    T->>T: CosyVoice合成
    T-->>C: 返回音频
    C->>V: 音频生成视频
    V->>V: ER-NeRF生成
    V-->>C: 返回视频
    C-->>F: 返回对话视频
    F->>U: 展示回复
```

### 技术架构栈

```mermaid
graph LR
    subgraph "前端技术 Frontend"
        A1[HTML5/CSS3]
        A2[JavaScript ES6]
        A3[Flask Templates]
    end

    subgraph "后端技术 Backend"
        B1[Flask 3.0.3]
        B2[Python 3.12]
        B3[Werkzeug]
        B4[Jinja2]
    end

    subgraph "AI框架 AI Frameworks"
        C1[PyTorch 2.3.1]
        C2[Transformers 4.51.3]
        C3[ONNX Runtime]
        C4[TensorRT 10.13]
    end

    subgraph "音频处理 Audio Processing"
        D1[Librosa 0.10.2]
        D2[SoundFile 0.12.1]
        D3[PyWorld 0.3.4]
    end

    subgraph "视频处理 Video Processing"
        E1[OpenCV]
        E2[FFmpeg]
        E3[PyAV]
    end

    subgraph "基础设施 Infrastructure"
        F1[Docker]
        F2[CUDA/cuDNN]
        F3[GPUtil]
        F4[psutil]
    end
```

### 部署架构

```mermaid
graph TB
    subgraph "用户侧 Client Side"
        U1[Web浏览器]
    end

    subgraph "应用服务器 Application Server"
        A1[Flask应用<br/>端口: 5001]
        A2[静态文件服务<br/>Static Files]
    end

    subgraph "AI服务集群 AI Services"
        S1[CosyVoice服务<br/>语音合成]
        S2[ER-NeRF服务<br/>Docker容器]
        S3[模型训练服务<br/>异步任务]
    end

    subgraph "存储层 Storage Layer"
        D1[(模型存储<br/>models/)]
        D2[(数据集存储<br/>datasets/)]
        D3[(结果缓存<br/>static/)]
    end

    subgraph "计算资源 Compute Resources"
        G1[GPU 0<br/>CUDA:0]
        G2[GPU 1<br/>CUDA:1]
    end

    U1 --> A1
    A1 --> A2
    A1 --> S1
    A1 --> S2
    A1 --> S3

    S1 --> D1
    S2 --> D1
    S3 --> D1
    S1 --> D2
    S2 --> D3

    S1 --> G1
    S2 --> G2
    S3 --> G1
```

