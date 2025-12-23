"""
音频预处理模块

架构：
    AudioPreprocessor (核心处理器)
        ├── AudioQualityAnalyzer (质量分析)
        ├── NoiseReducer (降噪策略)
        ├── AudioNormalizer (归一化)
        ├── SilenceRemover (静音去除)
        └── FormatConverter (格式转换)
"""

import os
import sys
import time
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path

# 导入现有的工具模块
from .path_manager import PathManager
import numpy as np

# 抑制 numba/debug 日志输出
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.WARNING)

# 音频处理库
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[警告] librosa 未安装，音频预处理功能将受限")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("[提示] noisereduce 未安装，降噪功能将使用基础算法")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


# ==================== 异常类层次结构 ====================

class AudioPreprocessError(Exception):
    """音频预处理基础异常"""
    pass


class AudioLoadError(AudioPreprocessError):
    """音频加载失败"""
    pass


class AudioValidationError(AudioPreprocessError):
    """音频验证失败"""
    pass


class PreprocessProcessingError(AudioPreprocessError):
    """预处理处理失败"""
    pass


# ==================== 配置类 ====================

class ProcessingMode(Enum):
    """处理模式"""
    QUALITY = "quality"        # 质量优先（慢但效果好）
    BALANCED = "balanced"      # 平衡模式
    FAST = "fast"              # 速度优先


@dataclass
class AudioMetadata:
    """音频元数据"""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int
    format: str
    bitrate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "file_path": self.file_path,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "file_size": self.file_size,
            "format": self.format,
            "bitrate": self.bitrate
        }


@dataclass
class PreprocessConfig:
    """
    预处理配置类

    Attributes:
        target_sample_rate: 目标采样率 (Hz)
        remove_silence: 是否去除静音
        silence_threshold: 静音阈值 (dB)
        normalize: 是否归一化音量
        denoise: 是否降噪
        enhance: 是否增强音频
        trim: 是否裁剪首尾静音
        mode: 处理模式 (quality/balanced/fast)
    """
    target_sample_rate: int = 24000
    remove_silence: bool = True
    silence_threshold: float = 20.0  # dB
    normalize: bool = True
    denoise: bool = True
    enhance: bool = False
    trim: bool = True
    mode: ProcessingMode = ProcessingMode.BALANCED

    # 验证范围
    MIN_SAMPLE_RATE = 8000
    MAX_SAMPLE_RATE = 48000
    MIN_DURATION = 1.0  # 秒
    MAX_DURATION = 60.0  # 秒

    def __post_init__(self):
        """配置验证"""
        if not self.MIN_SAMPLE_RATE <= self.target_sample_rate <= self.MAX_SAMPLE_RATE:
            raise ValueError(f"采样率必须在 {self.MIN_SAMPLE_RATE}-{self.MAX_SAMPLE_RATE} 范围内")


@dataclass
class PreprocessResult:
    """
    预处理结果类

    Attributes:
        success: 是否成功
        output_path: 输出文件路径
        input_metadata: 输入音频元数据
        output_metadata: 输出音频元数据
        processing_time: 处理耗时（秒）
        quality_score: 质量评分 (0-100)
        warnings: 警告信息列表
        error_message: 错误信息
        applied_steps: 应用的处理步骤
    """
    success: bool
    output_path: Optional[str] = None
    input_metadata: Optional[AudioMetadata] = None
    output_metadata: Optional[AudioMetadata] = None
    processing_time: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    applied_steps: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.success:
            return (
                f"PreprocessResult(success=True, "
                f"output='{os.path.basename(self.output_path)}', "
                f"quality={self.quality_score:.1f}, "
                f"time={self.processing_time:.3f}s)"
            )
        return f"PreprocessResult(success=False, error='{self.error_message}')"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "quality_score": self.quality_score,
            "processing_time": self.processing_time,
            "warnings": self.warnings,
            "error_message": self.error_message,
            "applied_steps": self.applied_steps,
            "input_metadata": self.input_metadata.to_dict() if self.input_metadata else None,
            "output_metadata": self.output_metadata.to_dict() if self.output_metadata else None
        }


# ==================== 策略模式接口 ====================

class PreprocessStrategy(ABC):
    """预处理策略抽象基类"""

    @abstractmethod
    def process(self, audio: np.ndarray, sr: int, config: PreprocessConfig) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        执行预处理

        Args:
            audio: 音频数据
            sr: 采样率
            config: 配置

        Returns:
            (处理后的音频, 采样率, 处理信息字典)

        Raises:
            PreprocessProcessingError: 处理失败时抛出
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


# ==================== 质量分析器 ====================

class AudioQualityAnalyzer:
    """
    音频质量分析器

    功能：
    - 评估音频是否适合语音克隆
    - 提供优化建议
    - 生成质量评分
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        分析音频质量

        Returns:
            质量分析结果字典
        """
        scores = {}

        # 1. 时长评分
        duration = len(audio) / sr
        scores['duration'] = self._score_duration(duration)
        scores['duration_value'] = duration

        # 2. 采样率评分
        scores['sample_rate'] = self._score_sample_rate(sr)
        scores['sample_rate_value'] = sr

        # 3. 清晰度评分 (零交叉率)
        scores['clarity'] = self._score_clarity(audio)

        # 4. 信噪比评分
        scores['snr'] = self._score_snr(audio)

        # 5. 能量评分
        scores['energy'] = self._score_energy(audio)

        # 6. 动态范围评分
        scores['dynamic_range'] = self._score_dynamic_range(audio)

        # 综合评分
        weights = {
            'duration': 0.20,
            'sample_rate': 0.15,
            'clarity': 0.25,
            'snr': 0.20,
            'energy': 0.10,
            'dynamic_range': 0.10
        }

        overall = sum(scores[key] * weights[key] for key in weights.keys())
        scores['overall'] = round(overall, 1)

        # 生成建议
        scores['recommendations'] = self._generate_recommendations(scores)

        return scores

    def _score_duration(self, duration: float) -> float:
        """评分：时长"""
        optimal_min, optimal_max = 5.0, 10.0

        if duration < optimal_min:
            score = (duration / optimal_min) * 60
        elif duration > optimal_max:
            excess = duration - optimal_max
            score = max(40, 100 - excess * 5)
        else:
            score = 100

        return min(100, max(0, score))

    def _score_sample_rate(self, sr: int) -> float:
        """评分：采样率"""
        if sr >= 24000:
            return 100
        elif sr >= 16000:
            return 85
        elif sr >= 8000:
            return 60
        else:
            return 30

    def _score_clarity(self, audio: np.ndarray) -> float:
        """评分：清晰度（基于零交叉率）"""
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)

        # 语音的 ZCR 通常在 0.1-0.5 之间
        if 0.1 <= mean_zcr <= 0.5:
            return 100
        elif 0.05 <= mean_zcr < 0.1 or 0.5 < mean_zcr <= 0.7:
            return 80
        else:
            return 60

    def _score_snr(self, audio: np.ndarray) -> float:
        """评分：信噪比（基于 RMS）"""
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > 0.15:
            return 100
        elif rms > 0.1:
            return 85
        elif rms > 0.05:
            return 70
        elif rms > 0.02:
            return 50
        else:
            return 30

    def _score_energy(self, audio: np.ndarray) -> float:
        """评分：能量分布"""
        energy = np.sum(audio ** 2)
        max_expected = len(audio) * 0.25
        score = min(100, (energy / max_expected) * 100)
        return max(30, score)

    def _score_dynamic_range(self, audio: np.ndarray) -> float:
        """评分：动态范围"""
        dynamic_range = np.max(audio) - np.min(audio)

        if 0.3 <= dynamic_range <= 0.8:
            return 100
        elif 0.2 <= dynamic_range < 0.3 or 0.8 < dynamic_range <= 0.95:
            return 80
        else:
            return 60

    def _generate_recommendations(self, scores: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 时长建议
        duration = scores.get('duration_value', 0)
        if duration < 5:
            recommendations.append("⚠️ 音频过短，建议使用 5-10 秒的音频")
        elif duration > 15:
            recommendations.append("⚠️ 音频过长，建议剪辑到 5-10 秒")

        # 采样率建议
        if scores.get('sample_rate', 0) < 80:
            recommendations.append("⚠️ 采样率过低，建议使用 16kHz 或更高")

        # 清晰度建议
        if scores.get('clarity', 0) < 70:
            recommendations.append("⚠️ 音频可能不够清晰，建议重新录制")

        # 能量建议
        if scores.get('energy', 0) < 60:
            recommendations.append("⚠️ 音频音量过低，建议提高音量")
        elif scores.get('energy', 0) > 95:
            recommendations.append("⚠️ 音频可能存在失真，建议降低音量")

        # 动态范围建议
        if scores.get('dynamic_range', 0) < 60:
            recommendations.append("⚠️ 音频动态范围过小，可能缺乏表现力")

        return recommendations


# ==================== 预处理策略实现 ====================

class ComprehensivePreprocessStrategy(PreprocessStrategy):
    """
    综合预处理策略

    按顺序执行：
    1. 裁剪静音
    2. 重采样
    3. 降噪
    4. 归一化
    5. 增强（可选）
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quality_analyzer = AudioQualityAnalyzer()

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        config: PreprocessConfig
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """执行综合预处理"""

        if not LIBROSA_AVAILABLE:
            raise PreprocessProcessingError("librosa 未安装，无法执行预处理")

        processed_audio = audio.copy()
        processed_sr = sr
        applied_steps = []
        info = {}

        try:
            # 1. 裁剪首尾静音
            if config.trim:
                processed_audio = self._trim_silence(processed_audio)
                applied_steps.append("trim_silence")

            # 2. 重采样
            if sr != config.target_sample_rate:
                processed_audio = librosa.resample(
                    processed_audio,
                    orig_sr=sr,
                    target_sr=config.target_sample_rate
                )
                processed_sr = config.target_sample_rate
                applied_steps.append("resample")
                info['original_sample_rate'] = sr
                info['target_sample_rate'] = config.target_sample_rate

            # 3. 降噪
            if config.denoise:
                processed_audio = self._denoise(processed_audio, processed_sr)
                applied_steps.append("denoise")

            # 4. 归一化
            if config.normalize:
                processed_audio = self._normalize(processed_audio)
                applied_steps.append("normalize")
                info['peak_db_before'] = float(20 * np.log10(np.max(np.abs(audio)) + 1e-10))
                info['peak_db_after'] = float(20 * np.log10(np.max(np.abs(processed_audio)) + 1e-10))

            # 5. 增强（可选）
            if config.enhance:
                processed_audio = self._enhance(processed_audio, processed_sr)
                applied_steps.append("enhance")

            # 质量分析
            quality_scores = self.quality_analyzer.analyze(processed_audio, processed_sr)
            info['quality_scores'] = quality_scores

            self.logger.info(f"预处理完成: {', '.join(applied_steps)}")

            return processed_audio, processed_sr, info

        except Exception as e:
            self.logger.error(f"预处理失败: {e}")
            raise PreprocessProcessingError(f"预处理失败: {e}") from e

    def _trim_silence(self, audio: np.ndarray, threshold_db: float = 20.0) -> np.ndarray:
        """裁剪首尾静音"""
        trimmed, _ = librosa.effects.trim(audio, top_db=threshold_db)

        if len(trimmed) < len(audio) * 0.5:
            # 裁剪后太短，保留原音频
            return audio

        return trimmed

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """降噪"""
        if NOISEREDUCE_AVAILABLE:
            try:
                # 使用 noisereduce 库
                # 取中间部分作为噪声样本（假设前后 0.5 秒是噪声）
                noise_sample = int(0.5 * sr)
                if len(audio) > noise_sample * 2:
                    noise = np.concatenate([audio[:noise_sample], audio[-noise_sample:]])
                    denoised = nr.reduce_noise(y=audio, y_noise=noise, sr=sr)
                else:
                    denoised = nr.reduce_noise(y=audio, sr=sr)

                return denoised
            except Exception as e:
                self.logger.warning(f"noisereduce 失败，使用基础降噪: {e}")

        # 基础降噪：使用谱减法的简化版本
        # 简单的高通滤波
        return self._basic_denoise(audio, sr)

    def _basic_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """基础降噪（高通滤波）"""
        from scipy import signal

        # 设计高通滤波器（截止频率 80Hz）
        nyquist = sr / 2
        cutoff = 80.0 / nyquist
        b, a = signal.butter(4, cutoff, btype='high')

        filtered = signal.filtfilt(b, a, audio)
        return filtered

    def _normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """音量归一化"""
        # 计算当前 RMS
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-10:
            return audio

        # 目标 RMS
        target_rms = 10 ** (target_db / 20)

        # 计算增益
        gain = target_rms / rms

        # 限制增益在合理范围
        gain = np.clip(gain, 0.5, 2.0)

        return audio * gain

    def _enhance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """音频增强"""
        # 简单的增强：轻微的动态范围压缩
        # 使用软限制
        threshold = 0.8
        ratio = 4.0

        enhanced = np.copy(audio)

        # 对超过阈值的部分应用压缩
        mask = np.abs(audio) > threshold
        enhanced[mask] = threshold + (audio[mask] - threshold) / ratio

        return enhanced

    def get_strategy_name(self) -> str:
        return "comprehensive"


# ==================== 核心预处理器 ====================

class AudioPreprocessor:
    """
    音频预处理器（核心类）

    职责：
    - 加载和验证音频
    - 执行预处理
    - 保存处理结果
    - 管理临时文件
    - 性能监控
    """

    def __init__(
        self,
        strategy: Optional[PreprocessStrategy] = None,
        output_dir: Optional[str] = None
    ):
        """
        初始化预处理器

        Args:
            strategy: 预处理策略（默认使用综合策略）
            output_dir: 输出目录（None 表示临时目录）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategy = strategy or ComprehensivePreprocessStrategy()
        self.quality_analyzer = AudioQualityAnalyzer()
        self.output_dir = output_dir or tempfile.gettempdir()

        # 性能统计
        self._stats = {
            'processed_count': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int, AudioMetadata]:
        """
        加载音频文件

        Returns:
            (音频数据, 采样率, 元数据)

        Raises:
            AudioLoadError: 加载失败
        """
        if not os.path.exists(file_path):
            raise AudioLoadError(f"文件不存在: {file_path}")

        if not LIBROSA_AVAILABLE:
            raise AudioLoadError("librosa 未安装，无法加载音频")

        try:
            # 加载音频
            audio, sr = librosa.load(file_path, sr=None)

            # 获取元数据
            file_size = os.path.getsize(file_path)
            metadata = AudioMetadata(
                file_path=file_path,
                duration=len(audio) / sr,
                sample_rate=sr,
                channels=1,  # librosa 默认转为单声道
                file_size=file_size,
                format='wav'
            )

            self.logger.info(f"加载音频: {file_path}")
            self.logger.info(f"  时长: {metadata.duration:.2f}s")
            self.logger.info(f"  采样率: {metadata.sample_rate}Hz")
            self.logger.info(f"  文件大小: {metadata.file_size / 1024:.1f}KB")

            return audio, sr, metadata

        except Exception as e:
            raise AudioLoadError(f"加载音频失败: {e}") from e

    def save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: str
    ) -> None:
        """
        保存音频文件

        Args:
            audio: 音频数据
            sr: 采样率
            output_path: 输出路径

        Raises:
            PreprocessProcessingError: 保存失败
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存音频
            sf.write(output_path, audio, sr)

            self.logger.info(f"保存音频: {output_path}")

        except Exception as e:
            raise PreprocessProcessingError(f"保存音频失败: {e}") from e

    def validate_audio(self, audio: np.ndarray, sr: int, config: PreprocessConfig) -> List[str]:
        """
        验证音频是否符合要求

        Returns:
            警告列表（空列表表示验证通过）
        """
        warnings = []

        # 检查采样率
        if sr < config.MIN_SAMPLE_RATE:
            warnings.append(f"采样率过低 ({sr}Hz < {config.MIN_SAMPLE_RATE}Hz)")

        # 检查时长
        duration = len(audio) / sr
        if duration < config.MIN_DURATION:
            warnings.append(f"时长过短 ({duration:.2f}s < {config.MIN_DURATION}s)")
        if duration > config.MAX_DURATION:
            warnings.append(f"时长过长 ({duration:.2f}s > {config.MAX_DURATION}s)")

        # 检查是否为静音
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            warnings.append("音频可能为静音")

        return warnings

    def preprocess(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        config: Optional[PreprocessConfig] = None
    ) -> PreprocessResult:
        """
        执行音频预处理

        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径（None 表示自动生成）
            config: 预处理配置（None 表示使用默认配置）

        Returns:
            PreprocessResult: 预处理结果
        """
        start_time = time.time()
        task_id = f"preprocess_{int(start_time * 1000)}"

        # 默认配置
        config = config or PreprocessConfig()

        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(
                self.output_dir,
                f"{base_name}_preprocessed_{int(start_time * 1000)}.wav"
            )

        result = PreprocessResult(success=False)

        try:
            self.logger.info(f"开始预处理: {input_path}")

            # 1. 加载音频
            audio, sr, input_metadata = self.load_audio(input_path)
            result.input_metadata = input_metadata

            # 2. 验证音频
            validation_warnings = self.validate_audio(audio, sr, config)
            result.warnings = validation_warnings

            if validation_warnings:
                self.logger.warning(f"音频验证警告: {'; '.join(validation_warnings)}")

            # 3. 执行预处理
            processed_audio, processed_sr, process_info = self.strategy.process(
                audio, sr, config
            )

            # 4. 再次验证处理后的音频
            post_warnings = self.validate_audio(processed_audio, processed_sr, config)
            if post_warnings:
                result.warnings.extend([f"处理后: {w}" for w in post_warnings])

            # 5. 保存结果
            self.save_audio(processed_audio, processed_sr, output_path)

            # 6. 生成输出元数据
            file_size = os.path.getsize(output_path)
            output_metadata = AudioMetadata(
                file_path=output_path,
                duration=len(processed_audio) / processed_sr,
                sample_rate=processed_sr,
                channels=1,
                file_size=file_size,
                format='wav'
            )

            # 7. 构建结果
            quality_scores = process_info.get('quality_scores', {})
            result = PreprocessResult(
                success=True,
                output_path=output_path,
                input_metadata=input_metadata,
                output_metadata=output_metadata,
                processing_time=time.time() - start_time,
                quality_score=quality_scores.get('overall', 0),
                warnings=result.warnings,
                applied_steps=process_info.get('quality_scores', {}).get('recommendations', [])
            )

            # 更新统计
            self._update_stats(time.time() - start_time, success=True)

            self.logger.info(f"预处理成功: {result}")

            return result

        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time

            # 更新统计
            self._update_stats(time.time() - start_time, success=False)

            self.logger.error(f"预处理失败: {e}")
            return result

    def _update_stats(self, duration: float, success: bool) -> None:
        """更新性能统计"""
        self._stats['processed_count'] += 1
        self._stats['total_time'] += duration
        if success:
            self._stats['success_count'] += 1
        else:
            self._stats['error_count'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self._stats.copy()
        if stats['processed_count'] > 0:
            stats['average_time'] = stats['total_time'] / stats['processed_count']
            stats['success_rate'] = stats['success_count'] / stats['processed_count']
        return stats


# ==================== 工厂模式 ====================

class AudioPreprocessorFactory:
    """预处理器工厂"""

    @staticmethod
    def create(
        mode: ProcessingMode = ProcessingMode.BALANCED,
        output_dir: Optional[str] = None
    ) -> AudioPreprocessor:
        """
        创建预处理器实例

        Args:
            mode: 处理模式
            output_dir: 输出目录

        Returns:
            AudioPreprocessor: 预处理器实例
        """
        strategy = ComprehensivePreprocessStrategy()
        return AudioPreprocessor(strategy=strategy, output_dir=output_dir)


# ==================== 便捷函数 ====================

def preprocess_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_sample_rate: int = 24000,
    denoise: bool = True,
    normalize: bool = True,
    remove_silence: bool = True
) -> PreprocessResult:
    """
    快捷预处理函数

    Args:
        input_path: 输入音频路径
        output_path: 输出路径（可选）
        target_sample_rate: 目标采样率
        denoise: 是否降噪
        normalize: 是否归一化
        remove_silence: 是否去除静音

    Returns:
        PreprocessResult: 预处理结果

    Example:
        >>> result = preprocess_audio("input.wav")
        >>> if result.success:
        ...     print(f"成功: {result.output_path}")
        ...     print(f"质量评分: {result.quality_score}")
    """
    config = PreprocessConfig(
        target_sample_rate=target_sample_rate,
        denoise=denoise,
        normalize=normalize,
        remove_silence=remove_silence
    )

    preprocessor = AudioPreprocessorFactory.create()
    return preprocessor.preprocess(input_path, output_path=output_path, config=config)


# ==================== 模块测试 ====================

if __name__ == "__main__":
    import sys

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    )

    print("=" * 70)
    print("音频预处理模块测试")
    print("=" * 70)

    # 测试配置
    try:
        config = PreprocessConfig(
            target_sample_rate=24000,
            denoise=True,
            normalize=True,
            trim=True
        )
        print(f"\n✓ 配置创建成功: target_sr={config.target_sample_rate}")
    except Exception as e:
        print(f"\n✗ 配置创建失败: {e}")
        sys.exit(1)

    # 测试工厂
    try:
        preprocessor = AudioPreprocessorFactory.create()
        print(f"✓ 预处理器创建成功")
    except Exception as e:
        print(f"✗ 预处理器创建失败: {e}")
        sys.exit(1)

    # 如果提供了音频文件，测试预处理
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\n测试预处理: {audio_file}")
            result = preprocess_audio(audio_file)
            print(f"\n结果: {result}")
            if result.success:
                print(f"  输出: {result.output_path}")
                print(f"  质量评分: {result.quality_score:.1f}/100")
                print(f"  处理时间: {result.processing_time:.3f}s")
                if result.warnings:
                    print(f"  警告: {'; '.join(result.warnings)}")
            else:
                print(f"  错误: {result.error_message}")
        else:
            print(f"\n音频文件不存在: {audio_file}")
    else:
        print(f"\n用法: python audio_preprocessor.py <audio_file>")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)