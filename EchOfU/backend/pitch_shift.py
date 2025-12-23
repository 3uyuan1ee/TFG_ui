"""
音频变调处理模块

"""

import os
import time
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

import librosa
import soundfile as sf


# ==================== 异常类层次结构 ====================

class PitchShiftError(Exception):
    """变调处理基础异常"""
    pass


class AudioLoadError(PitchShiftError):
    """音频加载失败"""
    pass


class AudioSaveError(PitchShiftError):
    """音频保存失败"""
    pass


class PitchShiftProcessingError(PitchShiftError):
    """变调处理失败"""
    pass


class InvalidParameterError(PitchShiftError):
    """无效参数"""
    pass


# ==================== 配置类 ====================

class PitchShiftUnit(Enum):
    """变调单位枚举"""
    SEMITONE = "semitone"      # 半音
    OCTAVE = "octave"          # 八度


@dataclass
class PitchShiftConfig:
    """
    变调配置类

    Attributes:
        pitch_steps: 变调步数（根据 unit 解释）
        unit: 变调单位（半音或八度）
        bins_per_octave: 每八度的 bin 数量（影响音质）
        n_fft: FFT 窗口大小（越大频率分辨率越高，但时间分辨率降低）
        hop_length: 跳跃长度（默认 n_fft // 4）
        preserve_duration: 是否保持音频时长（变调通常会影响时长）
        auto_normalize: 是否自动归一化音频
        quality_preset: 质量预设（快速/平衡/高质量）
    """
    pitch_steps: float
    unit: PitchShiftUnit = PitchShiftUnit.SEMITONE
    bins_per_octave: int = 12
    n_fft: int = 2048
    hop_length: Optional[int] = None
    preserve_duration: bool = True
    auto_normalize: bool = True
    quality_preset: str = "balanced"  # fast, balanced, high_quality

    # 验证范围
    PITCH_MIN: float = -24.0   # 最小变调（2个八度）
    PITCH_MAX: float = 24.0    # 最大变调（2个八度）

    def __post_init__(self):
        """配置验证"""
        # 验证变调范围
        if not self.PITCH_MIN <= self.pitch_steps <= self.PITCH_MAX:
            raise InvalidParameterError(
                f"变调值 {self.pitch_steps} 超出推荐范围 "
                f"[{self.PITCH_MIN}, {self.PITCH_MAX}]"
            )

        # 验证质量预设
        if self.quality_preset not in ("fast", "balanced", "high_quality"):
            raise InvalidParameterError(
                f"未知的质量预设: {self.quality_preset}"
            )

        # 根据质量预设调整参数
        self._apply_quality_preset()

        # 设置默认 hop_length
        if self.hop_length is None:
            self.hop_length = self.n_fft // 4

    def _apply_quality_preset(self):
        """根据质量预设调整参数"""
        presets = {
            "fast": {"n_fft": 1024, "bins_per_octave": 12},
            "balanced": {"n_fft": 2048, "bins_per_octave": 12},
            "high_quality": {"n_fft": 4096, "bins_per_octave": 24},
        }

        if self.quality_preset in presets:
            preset = presets[self.quality_preset]
            # 只在未显式设置的情况下应用预设
            if self.n_fft == 2048:  # 默认值
                self.n_fft = preset["n_fft"]
            if self.bins_per_octave == 12:  # 默认值
                self.bins_per_octave = preset["bins_per_octave"]

    def get_actual_steps(self) -> float:
        """获取实际的变调步数（转换为半音）"""
        if self.unit == PitchShiftUnit.OCTAVE:
            return self.pitch_steps * 12  # 1个八度 = 12半音
        return self.pitch_steps

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pitch_steps": self.pitch_steps,
            "unit": self.unit.value,
            "bins_per_octave": self.bins_per_octave,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "preserve_duration": self.preserve_duration,
            "auto_normalize": self.auto_normalize,
            "quality_preset": self.quality_preset,
        }


@dataclass
class PitchShiftResult:
    """
    变调处理结果类

    Attributes:
        success: 是否成功
        output_path: 输出文件路径
        input_path: 输入文件路径
        config: 使用的配置
        duration: 音频时长（秒）
        sample_rate: 采样率
        error_message: 错误信息（如果失败）
        processing_time: 处理耗时（秒）
        metadata: 额外元数据
    """
    success: bool
    output_path: Optional[str] = None
    input_path: Optional[str] = None
    config: Optional[PitchShiftConfig] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return (
                f"PitchShiftResult(success=True, "
                f"output='{os.path.basename(self.output_path)}', "
                f"pitch={self.config.pitch_steps if self.config else 'N/A'}, "
                f"time={self.processing_time:.3f}s)"
            )
        return f"PitchShiftResult(success=False, error='{self.error_message}')"


# ==================== 策略模式接口 ====================

class PitchShiftStrategy(ABC):
    """变调策略抽象基类"""

    @abstractmethod
    def shift_pitch(
        self,
        audio_data,
        sample_rate: int,
        config: PitchShiftConfig
    ) -> tuple:
        """
        执行变调处理

        Args:
            audio_data: 音频数据（numpy array）
            sample_rate: 采样率
            config: 变调配置

        Returns:
            tuple: (变调后的音频数据, 处理信息字典)

        Raises:
            PitchShiftProcessingError: 处理失败时抛出
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


class LibrosaPitchShiftStrategy(PitchShiftStrategy):
    """
    Librosa 变调策略实现

    使用 librosa.effects.pitch_shift 进行变调处理
    """

    def shift_pitch(
        self,
        audio_data,
        sample_rate: int,
        config: PitchShiftConfig
    ) -> tuple:
        """使用 Librosa 执行变调"""
        try:
            steps = config.get_actual_steps()

            # 执行变调
            shifted = librosa.effects.pitch_shift(
                audio_data,
                sr=sample_rate,
                n_steps=steps,
                bins_per_octave=config.bins_per_octave,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
            )

            # 自动归一化
            if config.auto_normalize:
                max_val = abs(shifted).max()
                if max_val > 0:
                    shifted = shifted / max_val

            # 处理信息
            info = {
                "strategy": self.get_strategy_name(),
                "actual_steps": steps,
                "fft_bins": config.n_fft,
                "hop_length": config.hop_length,
                "normalized": config.auto_normalize,
            }

            return shifted, info

        except Exception as e:
            raise PitchShiftProcessingError(f"Librosa 变调失败: {str(e)}") from e

    def get_strategy_name(self) -> str:
        return "librosa_pitch_shift"


# ==================== 临时文件管理器 ====================

class TempFileManager:
    """
    临时文件管理器

    职责：
    - 管理临时变调文件的生命周期
    - 自动清理过期文件
    - 线程安全的文件操作
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        max_age_hours: float = 24.0,
        cleanup_on_init: bool = True
    ):
        """
        初始化文件管理器

        Args:
            base_dir: 基础目录（None 表示使用系统临时目录）
            max_age_hours: 文件最大保留时间（小时）
            cleanup_on_init: 初始化时是否清理旧文件
        """
        self.base_dir = base_dir or tempfile.gettempdir()
        self.max_age_seconds = max_age_hours * 3600
        self.managed_files: List[str] = []

        if cleanup_on_init:
            self.cleanup_old_files()

    def generate_temp_path(
        self,
        prefix: str = "pitch_shift",
        suffix: str = ".wav"
    ) -> str:
        """
        生成临时文件路径

        Args:
            prefix: 文件名前缀
            suffix: 文件名后缀

        Returns:
            str: 临时文件路径
        """
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        filename = f"{prefix}_{timestamp}{suffix}"
        filepath = os.path.join(self.base_dir, filename)

        # 确保文件名唯一
        counter = 1
        while os.path.exists(filepath):
            filename = f"{prefix}_{timestamp}_{counter}{suffix}"
            filepath = os.path.join(self.base_dir, filename)
            counter += 1

        return filepath

    def register_file(self, filepath: str):
        """注册需要管理的文件"""
        if filepath not in self.managed_files:
            self.managed_files.append(filepath)

    def cleanup_file(self, filepath: str) -> bool:
        """
        清理指定文件

        Args:
            filepath: 文件路径

        Returns:
            bool: 是否成功删除
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                if filepath in self.managed_files:
                    self.managed_files.remove(filepath)
                return True
        except Exception as e:
            print(f"[TempFileManager] 清理文件失败 {filepath}: {e}")
        return False

    def cleanup_all_managed(self) -> int:
        """
        清理所有管理的文件

        Returns:
            int: 成功删除的文件数量
        """
        count = 0
        for filepath in self.managed_files[:]:  # 复制列表进行迭代
            if self.cleanup_file(filepath):
                count += 1
        return count

    def cleanup_old_files(self) -> int:
        """
        清理超过时效的临时文件

        Returns:
            int: 清理的文件数量
        """
        if not os.path.exists(self.base_dir):
            return 0

        current_time = time.time()
        count = 0

        try:
            for filename in os.listdir(self.base_dir):
                if filename.startswith("pitch_shift"):
                    filepath = os.path.join(self.base_dir, filename)
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > self.max_age_seconds:
                            if self.cleanup_file(filepath):
                                count += 1
        except Exception as e:
            print(f"[TempFileManager] 清理旧文件时出错: {e}")

        return count

    def __del__(self):
        """析构时清理管理的文件"""
        try:
            self.cleanup_all_managed()
        except:
            pass


# ==================== 核心处理器 ====================

class PitchShifter:
    """
    音频变调处理器

    职责：
    - 加载和保存音频
    - 执行变调处理
    - 管理处理流程
    """

    def __init__(
        self,
        strategy: Optional[PitchShiftStrategy] = None,
        file_manager: Optional[TempFileManager] = None
    ):
        """
        初始化变调处理器

        Args:
            strategy: 变调策略（默认使用 Librosa 策略）
            file_manager: 文件管理器（默认创建新实例）
        """
        self.strategy = strategy or LibrosaPitchShiftStrategy()
        self.file_manager = file_manager or TempFileManager()

    def load_audio(self, filepath: str) -> tuple:
        """
        加载音频文件

        Args:
            filepath: 音频文件路径

        Returns:
            tuple: (音频数据, 采样率)

        Raises:
            AudioLoadError: 加载失败
        """
        if not os.path.exists(filepath):
            raise AudioLoadError(f"文件不存在: {filepath}")

        try:
            audio_data, sample_rate = librosa.load(filepath, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            raise AudioLoadError(f"加载音频失败: {str(e)}") from e

    def save_audio(
        self,
        audio_data,
        sample_rate: int,
        filepath: str
    ) -> None:
        """
        保存音频文件

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            filepath: 保存路径

        Raises:
            AudioSaveError: 保存失败
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 保存音频
            sf.write(filepath, audio_data, sample_rate)
        except Exception as e:
            raise AudioSaveError(f"保存音频失败: {str(e)}") from e

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        config: Optional[PitchShiftConfig] = None,
        auto_cleanup: bool = True
    ) -> PitchShiftResult:
        """
        执行变调处理

        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径（None 表示自动生成）
            config: 变调配置（默认使用 pitch_steps=0）
            auto_cleanup: 失败时是否自动清理临时文件

        Returns:
            PitchShiftResult: 处理结果
        """
        start_time = time.time()

        # 默认配置（不变调）
        if config is None:
            config = PitchShiftConfig(pitch_steps=0)

        # 生成输出路径
        if output_path is None:
            pitch_value = config.pitch_steps
            output_path = self.file_manager.generate_temp_path(
                prefix=f"pitch_{pitch_value:.1f}"
            )

        # 注册输出文件以便管理
        self.file_manager.register_file(output_path)

        try:
            # 1. 加载音频
            audio_data, sample_rate = self.load_audio(input_path)
            duration = len(audio_data) / sample_rate

            # 2. 如果不需要变调，直接复制
            if config.pitch_steps == 0:
                shutil.copy2(input_path, output_path)
                return PitchShiftResult(
                    success=True,
                    output_path=output_path,
                    input_path=input_path,
                    config=config,
                    duration=duration,
                    sample_rate=sample_rate,
                    processing_time=time.time() - start_time,
                    metadata={"skipped": "No pitch shift needed"}
                )

            # 3. 执行变调
            shifted_audio, process_info = self.strategy.shift_pitch(
                audio_data, sample_rate, config
            )

            # 4. 保存结果
            self.save_audio(shifted_audio, sample_rate, output_path)

            return PitchShiftResult(
                success=True,
                output_path=output_path,
                input_path=input_path,
                config=config,
                duration=duration,
                sample_rate=sample_rate,
                processing_time=time.time() - start_time,
                metadata=process_info
            )

        except Exception as e:
            # 失败时清理
            if auto_cleanup and os.path.exists(output_path):
                self.file_manager.cleanup_file(output_path)

            return PitchShiftResult(
                success=False,
                input_path=input_path,
                config=config,
                error_message=str(e),
                processing_time=time.time() - start_time
            )


# ==================== 工厂模式 ====================

class PitchShifterFactory:
    """变调处理器工厂"""

    _strategies: Dict[str, PitchShiftStrategy] = {}

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy: PitchShiftStrategy
    ):
        """注册新策略"""
        cls._strategies[name] = strategy

    @classmethod
    def create(
        cls,
        strategy_name: str = "librosa",
        file_manager: Optional[TempFileManager] = None
    ) -> PitchShifter:
        """
        创建变调处理器

        Args:
            strategy_name: 策略名称
            file_manager: 文件管理器

        Returns:
            PitchShifter: 处理器实例
        """
        # 默认使用 Librosa 策略
        if strategy_name == "librosa" or strategy_name not in cls._strategies:
            strategy = LibrosaPitchShiftStrategy()
        else:
            strategy = cls._strategies[strategy_name]

        return PitchShifter(strategy=strategy, file_manager=file_manager)

    @classmethod
    def create_with_file_manager(
        cls,
        output_dir: str,
        strategy_name: str = "librosa",
        max_age_hours: float = 24.0
    ) -> PitchShifter:
        """
        创建带自定义文件管理器的处理器

        Args:
            output_dir: 输出目录
            strategy_name: 策略名称
            max_age_hours: 临时文件最大保留时间

        Returns:
            PitchShifter: 处理器实例
        """
        file_manager = TempFileManager(
            base_dir=output_dir,
            max_age_hours=max_age_hours
        )
        return cls.create(strategy_name, file_manager)


# ==================== 服务层（高层接口） ====================

class PitchShiftService:
    """
    变调服务（高层接口）

    提供简洁的 API，隐藏实现细节
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        auto_cleanup: bool = True
    ):
        """
        初始化服务

        Args:
            output_dir: 输出目录（None 表示使用临时目录）
            auto_cleanup: 是否自动清理
        """
        self.output_dir = output_dir
        self.auto_cleanup = auto_cleanup
        self.shifter = PitchShifterFactory.create_with_file_manager(
            output_dir or tempfile.gettempdir()
        )

    def shift_pitch(
        self,
        audio_path: str,
        pitch_steps: float,
        quality: str = "balanced"
    ) -> PitchShiftResult:
        """
        变调处理（简化接口）

        Args:
            audio_path: 音频文件路径
            pitch_steps: 变调步数（半音）
            quality: 质量预设（fast/balanced/high_quality）

        Returns:
            PitchShiftResult: 处理结果
        """
        config = PitchShiftConfig(
            pitch_steps=pitch_steps,
            quality_preset=quality
        )

        return self.shifter.process(
            input_path=audio_path,
            config=config,
            auto_cleanup=self.auto_cleanup
        )

    def cleanup_old_files(self, max_age_hours: float = 24.0) -> int:
        """清理旧文件"""
        return self.shifter.file_manager.cleanup_old_files()

    def __del__(self):
        """析构时清理"""
        if self.auto_cleanup:
            try:
                self.shifter.file_manager.cleanup_all_managed()
            except:
                pass


# ==================== 便捷函数 ====================

def shift_audio_pitch(
    audio_path: str,
    pitch_steps: float,
    output_path: Optional[str] = None,
    quality: str = "balanced"
) -> PitchShiftResult:
    """
    便捷函数：快速进行变调处理

    Args:
        audio_path: 输入音频路径
        pitch_steps: 变调步数（半音）
        output_path: 输出路径（None 表示自动生成）
        quality: 质量预设

    Returns:
        PitchShiftResult: 处理结果

    Example:
        >>> result = shift_audio_pitch("input.wav", pitch_steps=2.0)
        >>> if result.success:
        ...     print(f"成功: {result.output_path}")
    """
    service = PitchShiftService()
    config = PitchShiftConfig(pitch_steps=pitch_steps, quality_preset=quality)

    return service.shifter.process(
        input_path=audio_path,
        output_path=output_path,
        config=config
    )


# ==================== 模块测试 ====================

if __name__ == "__main__":
    # 简单测试
    import sys

    print("=" * 60)
    print("变调处理模块测试")
    print("=" * 60)

    # 测试配置
    try:
        config = PitchShiftConfig(pitch_steps=2.5, quality_preset="high_quality")
        print(f"\n✓ 配置创建成功: {config.to_dict()}")
    except Exception as e:
        print(f"\n✗ 配置创建失败: {e}")

    # 测试策略
    try:
        strategy = LibrosaPitchShiftStrategy()
        print(f"✓ 策略创建成功: {strategy.get_strategy_name()}")
    except Exception as e:
        print(f"✗ 策略创建失败: {e}")

    # 测试文件管理器
    try:
        manager = TempFileManager(max_age_hours=1)
        temp_path = manager.generate_temp_path()
        print(f"✓ 文件管理器创建成功")
        print(f"  临时路径: {temp_path}")
        manager.cleanup_all_managed()
    except Exception as e:
        print(f"✗ 文件管理器创建失败: {e}")

    # 测试处理器
    try:
        shifter = PitchShifterFactory.create()
        print(f"✓ 处理器创建成功")
    except Exception as e:
        print(f"✗ 处理器创建失败: {e}")

    # 如果提供了音频文件，测试处理
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\n测试处理音频: {audio_file}")
            result = shift_audio_pitch(audio_file, pitch_steps=2.0)
            print(f"结果: {result}")
            if result.success:
                print(f"  输出: {result.output_path}")
                print(f"  耗时: {result.processing_time:.3f}s")
        else:
            print(f"\n音频文件不存在: {audio_file}")
    else:
        print(f"\n用法: python pitch_shift.py <audio_file>")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)