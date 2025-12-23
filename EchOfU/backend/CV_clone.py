"""
CosyVoice3 语音克隆服务模块

核心功能:
- 基于CosyVoice3的零样本语音克隆
- 高质量语音合成与音频处理
- 参考音频管理与验证

"""

import os
import sys
import time
import uuid
import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torchaudio
import numpy as np

# =============================================================================
# 环境变量设置（解决中文路径问题）
# =============================================================================
# 设置 modelscope 缓存到项目目录，避免用户目录中文路径导致的问题
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "modelscope_cache"
os.environ['MODELSCOPE_CACHE'] = str(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# 添加CosyVoice路径
COSYVOICE_PATH = Path(__file__).parent.parent / "CosyVoice"
sys.path.append(str(COSYVOICE_PATH))

# 添加Matcha-TTS路径（CosyVoice的依赖）
MATCHA_TTS_PATH = COSYVOICE_PATH / "third_party" / "Matcha-TTS"
if MATCHA_TTS_PATH.exists():
    sys.path.insert(0, str(MATCHA_TTS_PATH))

# 导入现有的工具模块
from .path_manager import PathManager
from .model_download_manager import ModelDownloadManager, DownloadSource, ModelType

# CosyVoice导入检查
try:
    from cosyvoice.cli.cosyvoice import AutoModel
    COSYVOICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[CosyService] CosyVoice模块导入失败: {e}")
    COSYVOICE_AVAILABLE = False
    AutoModel = None


# ==================== 异常类定义 ====================

class CosyServiceError(Exception):
    """CosyService基础异常类"""
    pass


class ModelLoadError(CosyServiceError):
    """模型加载异常"""
    pass


class AudioValidationError(CosyServiceError):
    """音频验证异常"""
    pass


class VoiceGenerationError(CosyServiceError):
    """语音生成异常"""
    pass


class ConfigurationError(CosyServiceError):
    """配置异常"""
    pass


# ==================== 数据类定义 ====================

@dataclass
class AudioMetadata:
    """音频元数据"""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int
    created_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def is_valid(self) -> bool:
        """检查音频是否有效"""
        return (self.duration > 0 and
                self.sample_rate > 0 and
                os.path.exists(self.file_path))


@dataclass
class VoiceCloneRequest:
    """语音克隆请求"""
    text: str
    reference_audio_path: str
    prompt_text: Optional[str] = None
    output_filename: Optional[str] = None
    speed: float = 1.0
    stream: bool = False
    language: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """请求验证"""
        if not self.text or not self.text.strip():
            raise ValueError("文本内容不能为空")

        if len(self.text) > 5000:
            raise ValueError("文本长度不能超过5000字符")

        if not os.path.exists(self.reference_audio_path):
            raise FileNotFoundError(f"参考音频文件不存在: {self.reference_audio_path}")

        if self.speed <= 0 or self.speed > 3.0:
            raise ValueError("语速必须在0-3.0范围内")


@dataclass
class VoiceCloneResult:
    """语音克隆结果"""
    success: bool
    audio_path: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    generation_time: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_message: Optional[str] = None
    created_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return (self.success and
                self.audio_path and
                os.path.exists(self.audio_path) and
                self.audio_metadata is not None)


# ==================== 枚举类定义 ====================

class AudioFormat(Enum):
    """音频格式枚举"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class GenerationStatus(Enum):
    """生成状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== 工具类 ====================

class LoggerMixin:
    """日志混入类"""

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
        return self._logger


class DeviceManager:
    """设备管理器 - 负责计算设备检测和优化"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.device = self._detect_optimal_device()
            self._initialized = True

    def _detect_optimal_device(self) -> str:
        """检测最优计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {"device": self.device}

        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda
            })

        return info


# ==================== 核心功能模块 ====================

class AudioValidator(LoggerMixin):
    """音频验证器 - 负责音频文件的验证和处理"""

    def __init__(self, max_duration: float = 30.0, min_duration: float = 1.0):
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

    def validate_audio_file(self, audio_path: str) -> AudioMetadata:
        """验证音频文件并返回元数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                raise AudioValidationError(f"音频文件不存在: {audio_path}")

            # 检查文件格式
            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise AudioValidationError(f"不支持的音频格式: {file_ext}")

            # 获取音频信息
            try:
                # 对于支持的格式，先尝试使用torchaudio加载
                waveform, sample_rate = torchaudio.load(audio_path)

                # 从加载的数据中获取信息
                num_channels = waveform.shape[0] if waveform.dim() >= 2 else 1
                num_frames = waveform.shape[1] if waveform.dim() >= 2 else waveform.shape[0]
                duration = num_frames / sample_rate

                metadata = AudioMetadata(
                    file_path=os.path.abspath(audio_path),
                    duration=duration,
                    sample_rate=sample_rate,
                    channels=num_channels,
                    file_size=os.path.getsize(audio_path)
                )

            except Exception as e:
                # 如果torchaudio加载失败，尝试其他方法
                try:
                    # 使用mutagen获取音频元数据（如果可用）
                    try:
                        from mutagen import File
                        audio_file = File(audio_path)
                        if audio_file is not None:
                            duration = audio_file.info.length
                            sample_rate = int(getattr(audio_file.info, 'sample_rate', 44100))
                            num_channels = getattr(audio_file.info, 'channels', 2)

                            metadata = AudioMetadata(
                                file_path=os.path.abspath(audio_path),
                                duration=duration,
                                sample_rate=sample_rate,
                                channels=num_channels,
                                file_size=os.path.getsize(audio_path)
                            )
                        else:
                            raise Exception("mutagen无法解析音频文件")
                    except ImportError:
                        # 最后回退方案：估算信息
                        file_size = os.path.getsize(audio_path)
                        # 假设平均比特率为128kbps
                        estimated_duration = (file_size * 8) / (128000)

                        metadata = AudioMetadata(
                            file_path=os.path.abspath(audio_path),
                            duration=estimated_duration,
                            sample_rate=44100,  # 默认采样率
                            channels=2,  # 默认立体声
                            file_size=file_size
                        )
                        self.logger.warning(f"使用估算的音频信息: {audio_path}")

                except Exception as fallback_e:
                    raise AudioValidationError(f"音频文件读取失败: {e} (回退方案也失败: {fallback_e})")

            # 验证音频时长
            if metadata.duration < self.min_duration:
                raise AudioValidationError(f"音频时长过短: {metadata.duration:.2f}s < {self.min_duration}s")

            if metadata.duration > self.max_duration:
                raise AudioValidationError(f"音频时长过长: {metadata.duration:.2f}s > {self.max_duration}s")

            # 验证采样率
            if metadata.sample_rate < 16000:
                raise AudioValidationError(f"采样率过低: {metadata.sample_rate}Hz < 16000Hz")

            self.logger.info(f"[AudioValidator] 音频验证通过: {audio_path}")
            return metadata

        except AudioValidationError:
            raise
        except Exception as e:
            raise AudioValidationError(f"音频验证失败: {e}")


class ModelManager(LoggerMixin):
    """模型管理器 - 负责CosyVoice3模型的加载和管理"""

    def __init__(self, model_dir: str, device_manager: DeviceManager,
                 load_vllm: bool = False, load_jit: bool = False, load_trt: bool = False,
                 fp16: bool = False, trt_concurrent: int = 1):
        self.model_dir = model_dir
        self.device_manager = device_manager
        self.model = None
        self.model_info = {}
        self._load_vllm = load_vllm
        self._load_jit = load_jit
        self._load_trt = load_trt
        self._fp16 = fp16
        self._trt_concurrent = trt_concurrent
        self._lock = threading.RLock()
        self._load_model()

    def _load_model(self):
        """加载CosyVoice3模型"""
        if not COSYVOICE_AVAILABLE:
            raise ModelLoadError("CosyVoice模块不可用")

        with self._lock:
            try:
                self.logger.info(f"[ModelManager] 开始加载模型: {self.model_dir}")

                # 检查模型目录
                if not os.path.exists(self.model_dir):
                    raise ModelLoadError(f"模型目录不存在: {self.model_dir}")

                # 使用 PathManager 进行模型完整性检查
                path_manager = PathManager()
                is_complete, missing_files, error_msg = path_manager.check_cosyvoice3_model_integrity(self.model_dir)

                if not is_complete:
                    self.logger.error(f"[ModelManager] 模型完整性检查失败: {error_msg}")
                    self.logger.error("[ModelManager] 缺失的文件:")
                    for missing in missing_files:
                        self.logger.error(f"  - {missing}")

                    raise ModelLoadError(
                        f"模型文件不完整，无法加载。缺失 {len(missing_files)} 个必需文件。\n"
                        f"建议: 重新下载模型或检查模型目录。\n"
                        f"缺失文件列表:\n" + "\n".join(f"  - {m}" for m in missing_files)
                    )

                # 检测模型类型（CosyVoice/CosyVoice2/CosyVoice3）
                cosyvoice3_yaml = os.path.join(self.model_dir, 'cosyvoice3.yaml')
                cosyvoice2_yaml = os.path.join(self.model_dir, 'cosyvoice2.yaml')
                cosyvoice_yaml = os.path.join(self.model_dir, 'cosyvoice.yaml')

                # 根据模型类型构建参数
                # CosyVoice3 不支持 load_jit 参数
                if os.path.exists(cosyvoice3_yaml):
                    self.logger.info("[ModelManager] 检测到 CosyVoice3 模型")
                    load_params = {
                        "model_dir": self.model_dir,
                        "load_trt": self._load_trt,
                        "load_vllm": self._load_vllm,
                        "fp16": self._fp16
                    }
                    if self._load_trt:
                        load_params["trt_concurrent"] = self._trt_concurrent
                elif os.path.exists(cosyvoice2_yaml):
                    self.logger.info("[ModelManager] 检测到 CosyVoice2 模型")
                    load_params = {
                        "model_dir": self.model_dir,
                        "load_jit": self._load_jit,
                        "load_trt": self._load_trt,
                        "load_vllm": self._load_vllm,
                        "fp16": self._fp16
                    }
                    if self._load_trt:
                        load_params["trt_concurrent"] = self._trt_concurrent
                else:
                    self.logger.info("[ModelManager] 检测到 CosyVoice 模型")
                    load_params = {
                        "model_dir": self.model_dir,
                        "load_jit": self._load_jit,
                        "load_trt": self._load_trt,
                        "fp16": self._fp16
                    }
                    if self._load_trt:
                        load_params["trt_concurrent"] = self._trt_concurrent

                self.model = AutoModel(**load_params)

                # 记录使用的优化选项
                optimizations = []
                if self._load_vllm:
                    optimizations.append("VLLM")
                if self._load_jit:
                    optimizations.append("JIT")
                if self._load_trt:
                    optimizations.append("TensorRT")
                if self._fp16:
                    optimizations.append("FP16")

                if optimizations:
                    self.logger.info(f"[ModelManager] 启用优化: {', '.join(optimizations)}")
                if self._load_vllm:
                    self.logger.info("[ModelManager] VLLM加速已启用，推理速度将提升")

                # 获取模型信息
                self.model_info = {
                    "model_dir": self.model_dir,
                    "sample_rate": getattr(self.model, 'sample_rate', 24000),
                    "loaded_time": time.time(),
                    "device": self.device_manager.device,
                    "optimizations": {
                        "vllm_enabled": self._load_vllm,
                        "jit_enabled": self._load_jit,
                        "trt_enabled": self._load_trt,
                        "fp16_enabled": self._fp16,
                        "trt_concurrent": self._trt_concurrent if self._load_trt else None
                    },
                    "performance_stats": {
                        "inference_count": 0,
                        "total_inference_time": 0.0,
                        "average_inference_time": 0.0
                    }
                }

                self.logger.info(f"[ModelManager] 模型加载成功")
                self.logger.info(f"  采样率: {self.model_info['sample_rate']}Hz")
                self.logger.info(f"  设备: {self.model_info['device']}")

            except Exception as e:
                self.logger.error(f"[ModelManager] 模型加载失败: {e}")
                raise ModelLoadError(f"模型加载失败: {e}")

    def get_model(self):
        """获取模型实例"""
        with self._lock:
            if self.model is None:
                raise ModelLoadError("模型未加载")
            return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_info.copy()

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def update_performance_stats(self, inference_time: float):
        """更新性能统计信息"""
        with self._lock:
            stats = self.model_info["performance_stats"]
            stats["inference_count"] += 1
            stats["total_inference_time"] += inference_time
            stats["average_inference_time"] = stats["total_inference_time"] / stats["inference_count"]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self._lock:
            return self.model_info["performance_stats"].copy()

    def reset_performance_stats(self):
        """重置性能统计信息"""
        with self._lock:
            self.model_info["performance_stats"] = {
                "inference_count": 0,
                "total_inference_time": 0.0,
                "average_inference_time": 0.0
            }

    def get_optimization_info(self) -> Dict[str, Any]:
        """获取优化信息"""
        return self.model_info["optimizations"].copy()


class AudioProcessor(LoggerMixin):
    """音频处理器 - 负责音频文件的加载、保存和格式转换"""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self._conversion_cache = {}

    def _needs_conversion(self, audio_path: str) -> bool:
        """检查音频是否需要格式转换"""
        # 不支持的格式需要转换为 .wav
        # torchaudio 的 soundfile 后端不支持这些格式
        unsupported_formats = {'.m4a', '.aac', '.wma', '.mp4', '.mp3', '.flac', '.ogg'}
        return Path(audio_path).suffix.lower() in unsupported_formats

    def _convert_to_wav(self, audio_path: str) -> str:
        """
        将音频文件转换为 WAV 格式

        使用 pydub + ffmpeg 进行格式转换

        Args:
            audio_path: 原始音频文件路径

        Returns:
            str: 转换后的 WAV 文件路径
        """
        try:
            from pydub import AudioSegment

            audio_path = os.path.abspath(audio_path)
            file_name = Path(audio_path).stem
            file_hash = hash(audio_path + str(os.path.getmtime(audio_path)))

            # 检查缓存
            if file_hash in self._conversion_cache:
                cached_path = self._conversion_cache[file_hash]
                if os.path.exists(cached_path):
                    self.logger.info(f"[AudioProcessor] 使用缓存的转换文件: {cached_path}")
                    return cached_path

            # 创建转换缓存目录
            cache_dir = Path(self.path_manager.get_ref_voice_path()) / "__converted__"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 生成转换后的文件路径
            output_path = str(cache_dir / f"{file_name}_converted.wav")

            self.logger.info(f"[AudioProcessor] 开始转换音频格式: {Path(audio_path).suffix} -> .wav")

            # 使用 pydub 进行转换
            audio = AudioSegment.from_file(audio_path)

            # 导出为 WAV 格式
            audio.export(output_path, format="wav")

            # 缓存转换结果
            self._conversion_cache[file_hash] = output_path

            self.logger.info(f"[AudioProcessor] 音频转换完成: {output_path}")
            self.logger.info(f"  原始大小: {os.path.getsize(audio_path) / 1024:.2f} KB")
            self.logger.info(f"  转换后大小: {os.path.getsize(output_path) / 1024:.2f} KB")

            return output_path

        except ImportError:
            raise AudioValidationError(
                "音频格式转换需要 pydub 库。请安装: pip install pydub\n"
                "同时需要安装 ffmpeg: https://ffmpeg.org/download.html"
            )
        except Exception as e:
            raise AudioValidationError(f"音频格式转换失败: {e}")

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """加载音频文件（自动进行格式转换）"""
        try:
            # 检查是否需要转换
            if self._needs_conversion(audio_path):
                self.logger.info(f"[AudioProcessor] 检测到需要转换的音频格式: {Path(audio_path).suffix}")
                audio_path = self._convert_to_wav(audio_path)

            waveform, sample_rate = torchaudio.load(audio_path)

            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform, sample_rate

        except Exception as e:
            raise AudioValidationError(f"音频加载失败: {e}")

    def save_audio(self, audio_data: torch.Tensor, sample_rate: int, output_path: str) -> bool:
        """保存音频文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 保存音频
            torchaudio.save(output_path, audio_data, sample_rate)

            self.logger.info(f"[AudioProcessor] 音频已保存: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"[AudioProcessor] 音频保存失败: {e}")
            return False

    def get_output_path(self, filename: str = None, format: str = "wav") -> str:
        """生成输出文件路径"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cosyvoice_clone_{timestamp}.{format}"
        elif not filename.endswith(f'.{format}'):
            # 如果用户提供了文件名但没有扩展名，添加扩展名
            filename = f"{filename}.{format}"

        # 直接使用 get_res_voice_path 避免双重扩展名
        return self.path_manager.get_res_voice_path(filename)


class VoiceCloner(LoggerMixin):
    """语音克隆器 - 核心语音克隆逻辑"""

    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._whisper_model = None  # 延迟加载 whisper 模型

    def _load_whisper_model(self):
        """延迟加载 whisper 模型"""
        if self._whisper_model is None:
            try:
                import whisper
                self.logger.info("[VoiceCloner] 加载 Whisper 模型用于语音识别...")
                self._whisper_model = whisper.load_model("base")
                self.logger.info("[VoiceCloner] Whisper 模型加载成功")
            except Exception as e:
                self.logger.warning(f"[VoiceCloner] Whisper 模型加载失败: {e}")
                self._whisper_model = False
        return self._whisper_model

    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """使用 Whisper 从音频中提取文本"""
        try:
            model = self._load_whisper_model()
            if model is False:
                return None

            # 加载音频并转录
            result = model.transcribe(audio_path, language='zh')
            text = result['text'].strip()
            self.logger.info(f"[VoiceCloner] ASR 识别文本: {text}")
            return text if text else None
        except Exception as e:
            self.logger.warning(f"[VoiceCloner] ASR 识别失败: {e}")
            return None

    def clone_voice(self, request: VoiceCloneRequest) -> VoiceCloneResult:
        """执行语音克隆"""
        start_time = time.time()
        result = VoiceCloneResult(success=False)

        try:
            self.logger.info(f"[VoiceCloner] 开始语音克隆: {request.request_id}")
            self.logger.info(f"  文本长度: {len(request.text)} 字符")
            self.logger.info(f"  参考音频: {request.reference_audio_path}")

            # 获取模型
            model = self.model_manager.get_model()

            # 生成输出路径
            if request.output_filename:
                output_path = self.audio_processor.get_output_path(request.output_filename)
            else:
                output_path = self.audio_processor.get_output_path()

            # 检查参考音频是否需要格式转换（CosyVoice内部使用torchaudio.load，不支持.m4a）
            reference_audio = request.reference_audio_path
            if self.audio_processor._needs_conversion(reference_audio):
                self.logger.info(f"[VoiceCloner] 参考音频需要格式转换: {Path(reference_audio).suffix}")
                reference_audio = self.audio_processor._convert_to_wav(reference_audio)
                self.logger.info(f"[VoiceCloner] 使用转换后的音频: {reference_audio}")

            # 准备提示文本 - 使用CosyVoice3推荐的高级提示格式
            # 必需的前缀，用于防止提示词内容被读入音频
            prefix_prompt = "You are a helpful assistant.<|endofprompt|>"

            if request.prompt_text:
                # 用户提供了自定义提示词（应该是参考音频中说话的内容）
                prompt_text = f"{prefix_prompt} {request.prompt_text}"
            else:
                # 未提供 prompt_text，使用 ASR 从参考音频中提取文本
                self.logger.info("[VoiceCloner] 未提供 prompt_text，尝试从参考音频中提取文本...")
                transcribed_text = self._transcribe_audio(reference_audio)

                if transcribed_text:
                    # 使用识别出的文本作为 prompt
                    prompt_text = f"{prefix_prompt} {transcribed_text}"
                    self.logger.info(f"[VoiceCloner] 使用 ASR 识别文本作为 prompt")
                else:
                    # ASR 失败，回退到使用合成文本（虽然不完美，但比只有前缀好）
                    self.logger.warning("[VoiceCloner] ASR 识别失败，使用合成文本作为 prompt（可能效果不佳）")
                    prompt_text = f"{prefix_prompt} {request.text}"

            # 执行语音克隆
            self.logger.info("[VoiceCloner] 开始生成语音...")

            # 收集所有音频片段（CosyVoice 可能会分多次生成）
            audio_segments = []
            for i, audio_data in enumerate(model.inference_zero_shot(
                request.text,
                prompt_text,
                reference_audio,  # 使用转换后的音频路径
                stream=request.stream,
                speed=request.speed
            )):
                if 'tts_speech' in audio_data:
                    # 收集音频片段
                    audio_segments.append(audio_data['tts_speech'])
                    self.logger.debug(f"[VoiceCloner] 收集音频片段 {i+1}, 长度: {audio_data['tts_speech'].shape}")

            if not audio_segments:
                raise VoiceGenerationError("语音生成失败：未生成有效音频")

            # 拼接所有音频片段
            self.logger.info(f"[VoiceCloner] 拼接 {len(audio_segments)} 个音频片段...")
            combined_audio = torch.cat(audio_segments, dim=-1)

            # 保存拼接后的音频
            if self.audio_processor.save_audio(
                combined_audio,
                self.model_manager.model_info['sample_rate'],
                output_path
            ):
                result = VoiceCloneResult(
                    success=True,
                    audio_path=output_path,
                    generation_time=time.time() - start_time
                )

                # 获取音频元数据
                try:
                    info = torchaudio.info(output_path)
                    result.audio_metadata = AudioMetadata(
                        file_path=output_path,
                        duration=info.num_frames / info.sample_rate,
                        sample_rate=info.sample_rate,
                        channels=info.num_channels,
                        file_size=os.path.getsize(output_path)
                    )
                except Exception as e:
                    self.logger.warning(f"获取音频元数据失败: {e}")
            else:
                raise VoiceGenerationError("音频保存失败")

            self.logger.info(f"[VoiceCloner] 语音克隆成功: {output_path}")
            self.logger.info(f"  生成时长: {result.audio_metadata.duration:.2f}s")
            self.logger.info(f"  耗时: {result.generation_time:.2f}s")

            # 更新性能统计
            self.model_manager.update_performance_stats(result.generation_time)

            return result

        except Exception as e:
            result.error_message = str(e)
            result.generation_time = time.time() - start_time
            self.logger.error(f"[VoiceCloner] 语音克隆失败: {e}")
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            return result

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# ==================== 主服务类 ====================

class CosyService(LoggerMixin):
    """
    CosyVoice3语音克隆服务主类

    提供统一的API接口，整合所有功能模块，实现高质量的语音克隆服务。

    主要功能:
    - 零样本语音克隆
    - 音频文件验证和处理
    - 模型管理和加载
    - 结果输出和管理

    使用示例:
        service = CosyService()
        result = service.clone_voice(
            text="你好，这是测试语音。",
            reference_audio_path="/path/to/reference.wav"
        )
        if result.is_valid:
            print(f"克隆成功: {result.audio_path}")
    """

    def __init__(self, model_dir: str = None,
                 load_vllm: bool = False, load_jit: bool = False, load_trt: bool = False,
                 fp16: bool = False, trt_concurrent: int = 1):
        if not hasattr(self, '_initialized'):
            self._initialize_service(model_dir, load_vllm, load_jit, load_trt, fp16, trt_concurrent)

    def _initialize_service(self, model_dir: str = None,
                           load_vllm: bool = False, load_jit: bool = False, load_trt: bool = False,
                           fp16: bool = False, trt_concurrent: int = 1):
        """初始化服务组件"""
        try:
            self.logger.info("[CosyService] 初始化CosyVoice3语音克隆服务...")

            # 初始化基础组件
            self.path_manager = PathManager()
            self.device_manager = DeviceManager()
            self.audio_validator = AudioValidator()
            self.audio_processor = AudioProcessor(self.path_manager)

            # 初始化模型下载管理器
            self.model_download_manager = ModelDownloadManager(self.path_manager)

            # 只有在CosyVoice可用时才初始化模型相关组件
            if COSYVOICE_AVAILABLE:
                # 设置默认模型目录
                if model_dir is None:
                    model_dir = self.path_manager.get_cosyvoice3_2512_model_path()

                # 检查模型是否存在，如果不存在则尝试自动下载
                if not os.path.exists(model_dir):
                    self.logger.info("[CosyService] 模型目录不存在，尝试自动下载...")
                    self._auto_download_required_models()

                # 初始化模型相关模块
                if os.path.exists(model_dir):
                    try:
                        self.model_manager = ModelManager(
                            model_dir, self.device_manager,
                            load_vllm=load_vllm,
                            load_jit=load_jit,
                            load_trt=load_trt,
                            fp16=fp16,
                            trt_concurrent=trt_concurrent
                        )
                        self.voice_cloner = VoiceCloner(self.model_manager, self.audio_processor)
                    except ModelLoadError as e:
                        # 模型加载失败，检查是否是完整性问题
                        is_complete, missing_files, _ = self.path_manager.check_cosyvoice3_model_integrity(model_dir)

                        if not is_complete:
                            self.logger.error(f"[CosyService] 模型不完整，缺失 {len(missing_files)} 个文件")
                            self.logger.error("[CosyService] 将尝试自动重新下载模型...")

                            # 尝试重新下载
                            try:
                                self._auto_download_required_models()

                                # 下载后重新检查
                                is_complete_after, _, _ = self.path_manager.check_cosyvoice3_model_integrity(model_dir)

                                if is_complete_after:
                                    self.logger.info("[CosyService] 模型下载完成，重新加载...")
                                    self.model_manager = ModelManager(
                                        model_dir, self.device_manager,
                                        load_vllm=load_vllm,
                                        load_jit=load_jit,
                                        load_trt=load_trt,
                                        fp16=fp16,
                                        trt_concurrent=trt_concurrent
                                    )
                                    self.voice_cloner = VoiceCloner(self.model_manager, self.audio_processor)
                                else:
                                    self.logger.error("[CosyService] 模型下载后仍不完整，请手动检查")
                                    self.model_manager = None
                                    self.voice_cloner = None
                            except Exception as download_error:
                                self.logger.error(f"[CosyService] 自动下载失败: {download_error}")
                                self.model_manager = None
                                self.voice_cloner = None
                        else:
                            # 其他模型加载错误
                            self.logger.error(f"[CosyService] 模型加载失败: {e}")
                            self.model_manager = None
                            self.voice_cloner = None
                else:
                    self.model_manager = None
                    self.voice_cloner = None
                    self.logger.warning("[CosyService] 模型文件不存在，仅提供基础功能")
            else:
                self.model_manager = None
                self.voice_cloner = None
                self.logger.warning("[CosyService] CosyVoice模块不可用，仅提供基础音频验证功能")

            self._initialized = True

            # 输出初始化信息
            self.logger.info("[CosyService] 服务初始化完成")
            self.logger.info(f"  设备: {self.device_manager.get_device_info()}")

            if self.model_manager:
                self.logger.info(f"  模型: {self.model_manager.get_model_info()}")
            else:
                self.logger.info("  模型: 未加载（CosyVoice不可用）")

        except Exception as e:
            self.logger.error(f"[CosyService] 服务初始化失败: {e}")
            raise CosyServiceError(f"服务初始化失败: {e}")

    def clone_voice(self, text: str, reference_audio_path: str,
                   prompt_text: str = None, output_filename: str = None,
                   speed: float = 1.0, stream: bool = False) -> VoiceCloneResult:
        """
        语音克隆主接口

        Args:
            text: 要克隆的文本内容
            reference_audio_path: 参考音频文件路径
            prompt_text: 提示文本（可选，推荐使用高级提示格式）
            output_filename: 输出文件名（可选）
            speed: 语速控制（0.1-3.0）
            stream: 是否使用流式推理

        Returns:
            VoiceCloneResult: 克隆结果

        Raises:
            CosyServiceError: 服务相关异常
        """
        try:
            # 检查CosyVoice是否可用
            if not COSYVOICE_AVAILABLE or not self.voice_cloner:
                return VoiceCloneResult(
                    success=False,
                    error_message="CosyVoice模块不可用，无法进行语音克隆"
                )

            # 创建请求对象
            request = VoiceCloneRequest(
                text=text,
                reference_audio_path=reference_audio_path,
                prompt_text=prompt_text,
                output_filename=output_filename,
                speed=speed,
                stream=stream
            )

            # 验证参考音频
            self.audio_validator.validate_audio_file(reference_audio_path)

            # 音频预处理（提升克隆效果）
            preprocessed_audio_path = self._preprocess_reference_audio(reference_audio_path)
            if preprocessed_audio_path:
                self.logger.info(f"[CosyService] 使用预处理后的音频: {preprocessed_audio_path}")
                # 更新请求中的参考音频路径
                request.reference_audio_path = preprocessed_audio_path

            # 执行语音克隆
            result = self.voice_cloner.clone_voice(request)

            return result

        except Exception as e:
            self.logger.error(f"[CosyService] 语音克隆失败: {e}")
            return VoiceCloneResult(
                success=False,
                error_message=str(e)
            )

    def validate_reference_audio(self, audio_path: str) -> AudioMetadata:
        """
        验证参考音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            AudioMetadata: 音频元数据

        Raises:
            AudioValidationError: 音频验证异常
        """
        return self.audio_validator.validate_audio_file(audio_path)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        model_loaded = self.model_manager.is_model_loaded() if self.model_manager else False
        model_info = self.model_manager.get_model_info() if self.model_manager else None

        return {
            "service_initialized": self._initialized,
            "cosyvoice_available": COSYVOICE_AVAILABLE,
            "device_info": self.device_manager.get_device_info(),
            "model_info": model_info,
            "model_loaded": model_loaded
        }

    def cleanup(self):
        """清理服务资源"""
        try:
            if hasattr(self, 'voice_cloner'):
                self.voice_cloner.cleanup()

            self.logger.info("[CosyService] 服务资源清理完成")

        except Exception as e:
            self.logger.error(f"[CosyService] 服务清理失败: {e}")

    def _auto_download_required_models(self):
        """自动下载必需模型"""
        try:
            self.logger.info("[CosyService] 开始自动下载必需模型...")

            # 下载最关键的模型
            required_models = [ModelType.COSYVOICE3_2512]

            results = self.model_download_manager.download_models(
                required_models,
                source=DownloadSource.AUTO,
                force=False,
                install_deps=False
            )

            # 检查下载结果
            for model_type, success in results.items():
                if success:
                    self.logger.info(f"模型 {model_type.value} 下载成功")
                else:
                    self.logger.warning(f"模型 {model_type.value} 下载失败，将在使用时提供下载指引")

        except Exception as e:
            self.logger.warning(f"自动下载模型失败: {e}")

    def download_model(self, model_type: ModelType, source: DownloadSource = DownloadSource.AUTO) -> bool:
        """
        下载指定模型

        Args:
            model_type: 模型类型
            source: 下载源

        Returns:
            bool: 下载是否成功
        """
        return self.model_download_manager.download_model(model_type, source)

    def download_models(self, model_types: List[ModelType],
                        source: DownloadSource = DownloadSource.AUTO) -> Dict[ModelType, bool]:
        """
        下载多个模型

        Args:
            model_types: 模型类型列表
            source: 下载源

        Returns:
            Dict[ModelType, bool]: 下载结果
        """
        return self.model_download_manager.download_models(model_types, source)

    def get_download_status(self) -> Dict[str, Any]:
        """获取模型下载状态"""
        return self.model_download_manager.get_download_statistics()

    def is_model_downloaded(self, model_type: ModelType) -> bool:
        """检查模型是否已下载"""
        return self.model_download_manager.is_model_downloaded(model_type)

    def get_model_path(self, model_type: ModelType) -> Optional[str]:
        """获取模型本地路径"""
        return self.model_download_manager.get_model_path(model_type)

    def _preprocess_reference_audio(self, audio_path: str) -> Optional[str]:
        """
        预处理参考音频以提升克隆效果

        Args:
            audio_path: 原始音频路径

        Returns:
            预处理后的音频路径，失败时返回 None
        """
        try:
            from .audio_preprocessor import AudioPreprocessorFactory, PreprocessConfig, ProcessingMode

            self.logger.info(f"[CosyService] 开始预处理参考音频: {audio_path}")

            # 创建预处理器，使用 PathManager 管理输出路径
            preprocessor = AudioPreprocessorFactory.create(
                output_dir=self.path_manager.get_ref_voice_path()
            )

            # 配置预处理参数
            config = PreprocessConfig(
                target_sample_rate=24000,  # CosyVoice 最优采样率
                trim=True,                  # 去除首尾静音
                denoise=True,               # 降噪
                normalize=True,             # 归一化音量
                mode=ProcessingMode.BALANCED
            )

            # 执行预处理
            result = preprocessor.preprocess(audio_path, config=config)

            if result.success:
                self.logger.info(f"[CosyService] 预处理成功: {result.output_path}")
                self.logger.info(f"  质量评分: {result.quality_score:.1f}/100")
                self.logger.info(f"  处理时间: {result.processing_time:.3f}s")

                if result.warnings:
                    for warning in result.warnings:
                        self.logger.warning(f"  警告: {warning}")

                return result.output_path
            else:
                self.logger.warning(f"[CosyService] 预处理失败: {result.error_message}")
                self.logger.warning(f"[CosyService] 将使用原始音频: {audio_path}")
                return None

        except ImportError:
            self.logger.warning("[CosyService] audio_preprocessor 模块不可用，跳过预处理")
            return None
        except Exception as e:
            self.logger.error(f"[CosyService] 预处理异常: {e}")
            return None

    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        models_info = {}
        for model_type, model_info in self.model_download_manager.get_available_models().items():
            models_info[model_type.value] = {
                "description": model_info.description,
                "size_gb": model_info.size_gb,
                "downloaded": self.is_model_downloaded(model_type),
                "local_path": self.get_model_path(model_type),
                "required": model_info.required,
                "priority": model_info.priority
            }
        return models_info

    def prepare_cosyvoice_models(self, auto_download: bool = True) -> bool:
        """
        准备CosyVoice模型，包括检查和下载

        Args:
            auto_download: 是否自动下载缺失的模型

        Returns:
            bool: 准备是否成功
        """
        try:
            if not COSYVOICE_AVAILABLE:
                self.logger.warning("[CosyService] CosyVoice模块不可用，跳过模型准备")
                return False

            # 检查必需模型
            required_models = [ModelType.COSYVOICE3_2512]
            missing_models = [mt for mt in required_models if not self.is_model_downloaded(mt)]

            if not missing_models:
                self.logger.info("[CosyService] 所有必需模型已就绪")
                return True

            if auto_download:
                self.logger.info(f"[CosyService] 发现 {len(missing_models)} 个缺失模型，开始下载...")
                results = self.download_models(missing_models)

                all_success = all(results.values())
                if all_success:
                    self.logger.info("[CosyService] 模型下载完成，CosyVoice准备就绪")
                else:
                    self.logger.warning("[CosyService] 部分模型下载失败")

                return all_success
            else:
                self.logger.warning(f"[CosyService] 发现 {len(missing_models)} 个缺失模型，需要手动下载: {[mt.value for mt in missing_models]}")
                return False

        except Exception as e:
            self.logger.error(f"[CosyService] 模型准备失败: {e}")
            return False

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取综合服务状态（包含模型下载信息）"""
        status = self.get_service_status()

        # 添加模型下载信息
        if hasattr(self, 'model_download_manager'):
            download_stats = self.model_download_manager.get_download_statistics()
            status.update({
                "download_statistics": download_stats,
                "available_models": self.get_available_models(),
                "model_download_available": True
            })
        else:
            status.update({
                "download_statistics": None,
                "available_models": {},
                "model_download_available": False
            })

        # 添加路径信息
        status["paths"] = {
            "cosyvoice_root": self.path_manager.get_cosyvoice_path(),
            "cosyvoice_models": self.path_manager.get_cosyvoice_models_path(),
            "cosyvoice3_2512": self.path_manager.get_cosyvoice3_2512_model_path()
        }

        return status


# ==================== 便捷函数 ====================

def get_cosy_service(model_dir: str = None) -> CosyService:
    """获取CosyService实例"""
    return CosyService(model_dir)


def quick_clone(text: str, reference_audio_path: str,
                output_filename: str = None) -> VoiceCloneResult:
    """快速语音克隆"""
    service = get_cosy_service()
    return service.clone_voice(
        text=text,
        reference_audio_path=reference_audio_path,
        output_filename=output_filename
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 获取服务实例
        service = get_cosy_service()

        # 打印服务状态
        status = service.get_service_status()
        print("=" * 50)
        print("CosyVoice3 语音克隆服务状态")
        print("=" * 50)
        print(f"服务初始化: {status['service_initialized']}")
        print(f"CosyVoice可用: {status['cosyvoice_available']}")
        print(f"设备: {status['device_info']['device']}")
        if status['model_info']:
            print(f"模型采样率: {status['model_info']['sample_rate']}Hz")
            print(f"模型目录: {status['model_info']['model_dir']}")
        print("=" * 50)

        # 示例使用（需要实际的音频文件）
        # result = service.clone_voice(
        #     text="你好，这是CosyVoice3语音克隆测试。",
        #     reference_audio_path="/path/to/reference/audio.wav",
        #     prompt_text="You are a helpful assistant. Please speak the following naturally.<|endofprompt|>",
        #     output_filename="test_clone.wav"
        # )
        #
        # if result.is_valid:
        #     print(f"克隆成功!")
        #     print(f"输出文件: {result.audio_path}")
        #     print(f"音频时长: {result.audio_metadata.duration:.2f}秒")
        #     print(f"生成耗时: {result.generation_time:.2f}秒")
        # else:
        #     print(f"克隆失败: {result.error_message}")

        print("\n服务已就绪，可以开始语音克隆！")

    except Exception as e:
        print(f"服务启动失败: {e}")
        traceback.print_exc()