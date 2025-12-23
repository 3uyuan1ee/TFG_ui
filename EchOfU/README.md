# EchOfU

基于 CosyVoice3 的语音克隆与合成系统，支持零样本语音克隆、多语言语音合成等功能。

## 环境要求

- **Python**: 3.10
- **GPU**: NVIDIA GPU（推荐，>=8GB显存）
- **OS**: Linux/macOS/Windows

## 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/3uyuan1ee/TFG_ui.git
cd TFG_ui
git submodule update --init --recursive
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
cd EchOfU
pip install -r requirements.txt
winget install ffmpeg

```

**国内用户加速**:
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 4. 下载模型

**方法1: 自动下载（运行时会自动下载）**
如果要使用自动下载，请务必关闭flask的调试模式！

**方法2: 手动下载**
```python
from modelscope import snapshot_download

# 下载 CosyVoice3-2512 模型
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                   local_dir='EchOfU/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')
```

### 5. 启动服务

使用IDE可能会导入失败，要将项目根设置为EchOfU
```bash
cd EchOfU
python app.py
```

访问: http://localhost:5001

## 快速使用

### Web界面

1. 打开浏览器访问 http://localhost:5001
2. 进入"音频克隆"页面
3. 上传参考音频（或录制）
4. 输入要合成的文本
5. 点击生成

### Python API

```python
from backend.CV_clone import CosyService

# 创建服务
service = CosyService()

# 语音克隆
result = service.clone_voice(
    text="你好，这是测试语音。",
    reference_audio_path="path/to/reference.wav"
)

if result.is_valid:
    print(f"生成成功: {result.audio_path}")
else:
    print(f"生成失败: {result.error_message}")
```

### 启用加速

```python
# VLLM 加速（推荐，需要GPU）
service = CosyService(load_vllm=True)

# VLLM + FP16（最佳性能）
service = CosyService(load_vllm=True, fp16=True)
```

## 模型下载

### ModelScope（国内推荐）

```python
from modelscope import snapshot_download

# 主要模型
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                   local_dir='EchOfU/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')

# 可选模型
snapshot_download('iic/CosyVoice2-0.5B',
                   local_dir='EchOfU/CosyVoice/pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M',
                   local_dir='EchOfU/CosyVoice/pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-ttsfrd',
                   local_dir='EchOfU/CosyVoice/pretrained_models/CosyVoice-ttsfrd')
```

### HuggingFace（海外用户）

```python
from huggingface_hub import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='EchOfU/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')
```

## 常见问题

### Q: 编译错误（grpcio等）

```bash
# 使用预编译版本
pip install --no-build-isolation grpcio==1.57.0 grpcio-tools==1.57.0
```

### Q: 子模块初始化失败

```bash
cd TFG_ui
git submodule update --init --recursive --depth=1
```

### Q: 内存不足

```python
# 使用小模型或禁用VLLM
service = CosyService(load_vllm=False)
```

### Q: macOS 编译问题

```bash
# 安装系统依赖
brew install cmake sox
xcode-select --install
```

## 目录结构

```
EchOfU/
├── app.py                 # Flask 应用入口
├── requirements.txt       # 所有依赖（已包含 CosyVoice + Matcha-TTS）
├── backend/              # 后端模块
│   ├── CV_clone.py      # CosyVoice 服务
│   └── ...
├── static/              # 静态资源
│   ├── voices/         # 音频文件
│   └── videos/         # 视频文件
└── CosyVoice/          # CosyVoice 子模块
    └── pretrained_models/  # 模型存放目录
```

## 性能优化

| 优化方式 | 速度提升 | 显存需求 |
|---------|---------|---------|
| 标准模式 | 1x | ~4GB |
| VLLM | 3-5x | ~8GB |
| VLLM + FP16 | 4-6x | ~6GB |

## License

MIT License