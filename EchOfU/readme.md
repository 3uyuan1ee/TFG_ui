### 1. 克隆仓库

```bash
git clone https://github.com/3uyuan1ee/TFG_ui.git
cd TFG_ui
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
cd EchOfU
pip install -r requirements.txt
```

### 3. 下载模型

```python
from backend.CV_clone import CosyService, ModelType

# 创建服务实例（会自动下载模型）
service = CosyService()

# 或手动下载模型
service = CosyService()
service.download_model(ModelType.COSYVOICE3_2512)
```

## 基本使用

```python
from backend.CV_clone import CosyService

# 创建服务实例
service = CosyService()

# 语音克隆
result = service.clone_voice(
    text="你好，这是测试语音。",
    reference_audio_path="path/to/reference.wav"
)

if result.is_valid:
    print(f"克隆成功: {result.audio_path}")
else:
    print(f"克隆失败: {result.error_message}")
```

## 高级用法

### VLLM加速（推荐）

VLLM可以显著提升CosyVoice的推理速度，特别适合批量处理和高频使用场景：

```python
from backend.CV_clone import CosyService

# 启用VLLM加速（需要GPU）
service = CosyService(load_vllm=True)

# VLLM + FP16组合优化（最佳性能）
service = CosyService(
    load_vllm=True,
    fp16=True
)

# 完整优化配置（VLLM + TensorRT + FP16）
service = CosyService(
    load_vllm=True,
    load_trt=True,        # 需要TensorRT环境
    fp16=True,
    trt_concurrent=4     # TensorRT并发数
)
```

**VLLM性能对比**：
- 标准模式：~2-5秒/句
- VLLM模式：~0.5-1秒/句（提升3-5倍）
- VLLM + FP16：~0.3-0.8秒/句（再提升20-30%）

### 性能优化选项

```python
from backend.CV_clone import CosyService

# 启用JIT编译（CPU推理优化）
service = CosyService(load_jit=True)

# 启用FP16（减少内存使用）
service = CosyService(fp16=True)

# 启用TensorRT（需要TensorRT环境，Linux）
service = CosyService(load_trt=True, fp16=True, trt_concurrent=4)

# 组合使用多种优化
service = CosyService(
    load_jit=True,
    fp16=True
)
```

### 性能监控

```python
from backend.CV_clone import CosyService

# 创建带VLLM优化的服务
service = CosyService(load_vllm=True)

# 获取性能统计
if service.model_manager:
    perf_stats = service.model_manager.get_performance_stats()
    print(f"推理次数: {perf_stats['inference_count']}")
    print(f"平均推理时间: {perf_stats['average_inference_time']:.2f}秒")

    # 获取优化信息
    opt_info = service.model_manager.get_optimization_info()
    print(f"VLLM启用: {opt_info['vllm_enabled']}")
    print(f"FP16启用: {opt_info['fp16_enabled']}")
    print(f"JIT启用: {opt_info['jit_enabled']}")
    print(f"TensorRT启用: {opt_info['trt_enabled']}")

    # 重置统计信息
    service.model_manager.reset_performance_stats()
```

### 自定义模型路径

```python
from backend.CV_clone import CosyService

# 使用自定义模型路径
service = CosyService(
    model_dir="/path/to/your/model",
    load_vllm=True
)
```

## VLLM安装说明

VLLM需要额外的依赖项，请确保：

```bash
# 安装VLLM
pip install vllm>=0.4.0

# 对于CUDA支持（推荐）
pip install vllm[cuda]

# 对于特定CUDA版本
pip install vllm --index-url https://download.pytorch.org/whl/cu118
```

**注意事项**：
- VLLM需要NVIDIA GPU和CUDA支持
- 建议GPU内存 >= 8GB
- 在CPU环境下，VLLM会自动回退到标准模式
- TensorRT需要Linux环境和额外配置