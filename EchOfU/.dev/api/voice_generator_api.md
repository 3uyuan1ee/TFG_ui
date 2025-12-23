# CosyVoice语音生成服务API

## 创建实例

### 方法1：使用默认配置
```python
from backend.voice_generator import get_voice_service
service = get_voice_service()
```

### 方法2：自定义配置
```python
from backend.voice_generator import CosyVoiceService, ServiceConfig

config = ServiceConfig(enable_vllm=True)  # 启用VLLM加速
service = CosyVoiceService(config)
```

## 主要方法

### 1. clone_voice - 语音克隆

**必须参数：**
- `text` (str): 要生成的文本
- `reference_audio` (str): 参考音频文件路径

**可选参数：**
- `prompt_text` (str): 提示文本
- `output_filename` (str): 输出文件名
- `speed` (float): 语速控制 (0.1-3.0, 默认1.0)
- `language` (Language): 语言 (默认中文)

**返回：**
- `VoiceGenerationResult`: 生成结果

```python
result = service.clone_voice(
    text="你好，这是测试。",
    reference_audio="reference.wav",
    speed=1.2,
    output_filename="output.wav"
)

if result.is_success:
    print(f"成功: {result.audio_path}")
else:
    print(f"失败: {result.error_message}")
```

### 2. generate_speech - 标准语音生成

**必须参数：**
- `text` (str): 要生成的文本

**可选参数：**
- `language` (Language): 语言 (默认中文)
- `output_filename` (str): 输出文件名

**注意：** 当前版本暂不支持，主要基于参考音频的语音克隆

```python
result = service.generate_speech("你好世界")
```

### 3. get_service_status - 获取服务状态

**参数：** 无

**返回：**
- `dict`: 服务状态信息

```python
status = service.get_service_status()
print(f"CosyVoice可用: {status['cosyvoice_available']}")
```

### 4. cleanup - 清理资源

**参数：** 无

**返回：** 无

```python
service.cleanup()
```

## 便捷函数

### quick_clone_voice - 快速语音克隆

**必须参数：**
- `text` (str): 文本
- `reference_audio` (str): 参考音频路径

**可选参数：**
- `output_filename` (str): 输出文件名
- `enable_vllm` (bool): 启用VLLM加速 (默认False)

```python
from backend.voice_generator import quick_clone_voice

result = quick_clone_voice("你好世界", "reference.wav")
```

### clone_voice_with_vllm - VLLM加速克隆

**必须参数：**
- `text` (str): 文本
- `reference_audio` (str): 参考音频路径

**可选参数：**
- `output_filename` (str): 输出文件名

```python
from backend.voice_generator import clone_voice_with_vllm

result = clone_voice_with_vllm("Hello World", "reference.wav")
```

## 配置选项

### ServiceConfig

- `enable_vllm` (bool): 启用VLLM加速 (默认False)
- `log_level` (str): 日志级别 (默认"INFO")

## 语言选项

```python
from backend.voice_generator import Language

Language.CHINESE  # 中文
Language.ENGLISH   # 英文
Language.JAPANESE  # 日文
Language.KOREAN    # 韩文
Language.AUTO      # 自动检测
```

## 结果对象

### VoiceGenerationResult

- `task_id` (str): 任务ID
- `success` (bool): 是否成功
- `audio_path` (str): 音频文件路径 (输出到 `EchOfU/static/voices/res_voices/`)
- `generation_time` (float): 生成时间
- `error_message` (str): 错误信息
- `is_success` (bool): 成功状态
- `is_failed` (bool): 失败状态

```python
if result.is_success:
    print(f"✅ 成功: {result.audio_path}")
    print(f"📁 输出目录: EchOfU/static/voices/res_voices/")
    print(f"⏱️ 耗时: {result.generation_time:.2f}秒")
else:
    print(f"❌ 失败: {result.error_message}")
```

## 上下文管理器（推荐）

```python
with get_voice_service() as service:
    result = service.clone_voice("测试", "reference.wav")
    # 自动清理资源
```