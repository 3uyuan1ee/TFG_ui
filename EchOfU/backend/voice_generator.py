"""
CosyVoice语音生成服务模块
提供简单、高质量的语音克隆服务

主要功能:
- 基于CosyVoice3的零样本语音克隆
- 多语言支持（中文、英文、日文、韩文等）
- 高性能VLLM加速
- 基础的音频处理和验证

设计原则:
- 简单易用
- 高质量
- 稳定可靠

"""

import os
import sys
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入CosyVoice核心模块
from .CV_clone import (
    VoiceCloneResult, AudioMetadata,
    get_cosy_service
)


# ==================== 枚举类定义 ====================

class Language(Enum):
    """支持的语言"""
    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    AUTO = "auto"          # 自动检测


# ==================== 数据类定义 ====================

@dataclass
class VoiceGenerationResult:
    """语音生成结果"""
    task_id: str
    success: bool
    audio_path: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    generation_time: float = 0.0
    error_message: Optional[str] = None
    created_time: datetime = None

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.task_id == "":
            self.task_id = str(uuid.uuid4())

    @property
    def is_success(self) -> bool:
        return self.success and self.audio_path is not None

    @property
    def is_failed(self) -> bool:
        return not self.success


@dataclass
class ServiceConfig:
    """服务配置"""
    enable_vllm: bool = False
    log_level: str = "INFO"


# ==================== 异常类定义 ====================

class VoiceGeneratorError(Exception):
    """语音生成器基础异常"""
    pass


class ServiceNotInitialized(VoiceGeneratorError):
    """服务未初始化异常"""
    pass


# ==================== 主服务类 ====================

class CosyVoiceService:
    """
    CosyVoice语音生成服务（简化版）

    提供简单直接的语音克隆功能：
    - 零样本语音克隆
    - VLLM加速
    - 多语言支持

    使用示例:
        # 创建服务实例
        service = CosyVoiceService(enable_vllm=True)

        # 语音克隆
        result = service.clone_voice(
            text="你好，这是测试。",
            reference_audio="path/to/reference.wav"
        )

        if result.is_success:
            print(f"克隆成功: {result.audio_path}")
    """

    _instance = None
    _lock = None

    def __new__(cls, config: ServiceConfig = None):
        """单例模式"""
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()

            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: ServiceConfig = None):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._config = config or ServiceConfig()
            self._initialize_service()
            self._initialized = True

    def _initialize_service(self):
        """初始化服务"""
        try:
            # 设置日志
            self._setup_logging()

            # 初始化CosyVoice核心服务
            self.logger.info("[CosyVoiceService] 初始化语音生成服务...")
            self._cosy_service = get_cosy_service()

            self.logger.info("[CosyVoiceService] 服务初始化完成")
            self._log_service_status()

        except Exception as e:
            self.logger.error(f"[CosyVoiceService] 服务初始化失败: {e}")
            raise ServiceNotInitialized(f"服务初始化失败: {e}")

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger("CosyVoiceService")
        self.logger.setLevel(getattr(logging, self._config.log_level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_service_status(self):
        """记录服务状态"""
        try:
            status = self.get_service_status()
            self.logger.info("=" * 50)
            self.logger.info("CosyVoice语音生成服务状态")
            self.logger.info("=" * 50)

            if status['cosyvoice_available']:
                self.logger.info("✅ CosyVoice核心服务: 已就绪")
                if status.get('model_info'):
                    self.logger.info(f"   模型: {status['model_info'].get('model_dir', 'N/A')}")
            else:
                self.logger.warning("❌ CosyVoice核心服务: 未就绪")

            self.logger.info(f"VLLM启用: {self._config.enable_vllm}")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.warning(f"记录服务状态失败: {e}")

    def clone_voice(self, text: str, reference_audio: str,
                   prompt_text: Optional[str] = None,
                   output_filename: Optional[str] = None,
                   speed: float = 1.0, language: Language = Language.CHINESE) -> VoiceGenerationResult:
        """
        语音克隆主接口

        Args:
            text: 要生成的文本内容
            reference_audio: 参考音频文件路径
            prompt_text: 提示文本（可选）
            output_filename: 输出文件名（可选）
            speed: 语速控制（0.1-3.0）
            language: 语言设置

        Returns:
            VoiceGenerationResult: 生成结果
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())

        result = VoiceGenerationResult(
            task_id=task_id,
            success=False
        )

        try:
            self.logger.info(f"[CosyVoiceService] 开始语音克隆: {task_id}")
            self.logger.info(f"  文本: {text[:50]}...")
            self.logger.info(f"  参考音频: {reference_audio}")

            # 输入验证
            if not text or not text.strip():
                result.error_message = "文本内容不能为空"
                return result

            if not os.path.exists(reference_audio):
                result.error_message = f"参考音频文件不存在: {reference_audio}"
                return result

            text = text.strip()
            if len(text) > 5000:
                result.error_message = "文本长度不能超过5000字符"
                return result

            if speed <= 0 or speed > 3.0:
                result.error_message = "语速必须在0-3.0范围内"
                return result

            # 执行语音生成
            cosy_result = self._cosy_service.clone_voice(
                text=text,
                reference_audio_path=reference_audio,
                prompt_text=prompt_text,
                output_filename=output_filename,
                speed=speed,
                stream=False
            )

            generation_time = time.time() - start_time

            if cosy_result.is_valid:
                result = VoiceGenerationResult(
                    task_id=task_id,
                    success=True,
                    audio_path=cosy_result.audio_path,
                    audio_metadata=cosy_result.audio_metadata,
                    generation_time=generation_time
                )

                self.logger.info(f"[CosyVoiceService] 语音克隆成功: {cosy_result.audio_path}")
                self.logger.info(f"  生成时长: {cosy_result.audio_metadata.duration:.2f}s")
                self.logger.info(f"  耗时: {generation_time:.2f}s")

            else:
                result.error_message = cosy_result.error_message or "生成失败"
                result.generation_time = generation_time
                self.logger.error(f"[CosyVoiceService] 语音克隆失败: {result.error_message}")

            return result

        except Exception as e:
            result.error_message = str(e)
            result.generation_time = time.time() - start_time
            self.logger.error(f"[CosyVoiceService] 语音克隆异常: {e}")
            return result

    def generate_speech(self, text: str, language: Language = Language.CHINESE,
                       output_filename: Optional[str] = None) -> VoiceGenerationResult:
        """
        标准语音生成（无需参考音频）

        注意：当前版本主要基于参考音频的语音克隆，此方法暂不支持

        Args:
            text: 要生成的文本
            language: 语言设置
            output_filename: 输出文件名

        Returns:
            VoiceGenerationResult: 生成结果
        """
        result = VoiceGenerationResult(
            task_id=str(uuid.uuid4()),
            success=False,
            error_message="当前版本主要基于参考音频进行语音克隆，此功能暂不支持"
        )
        return result

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        try:
            cosy_status = self._cosy_service.get_comprehensive_status()

            return {
                "service": {
                    "initialized": self._initialized,
                    "config": {
                        "enable_vllm": self._config.enable_vllm,
                        "log_level": self._config.log_level
                    }
                },
                "cosyvoice": cosy_status,
                "cosyvoice_available": cosy_status.get('cosyvoice_available', False)
            }
        except Exception as e:
            return {
                "service": {
                    "initialized": self._initialized,
                    "error": str(e)
                },
                "cosyvoice_available": False
            }

    def cleanup(self):
        """清理服务资源"""
        try:
            self.logger.info("[CosyVoiceService] 开始清理服务资源...")

            # 清理CosyVoice服务
            if hasattr(self, '_cosy_service'):
                self._cosy_service.cleanup()

            self.logger.info("[CosyVoiceService] 服务资源清理完成")

        except Exception as e:
            self.logger.error(f"[CosyVoiceService] 服务清理失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass


# ==================== 便捷函数 ====================

def get_voice_service(config: ServiceConfig = None) -> CosyVoiceService:
    """获取语音服务实例"""
    return CosyVoiceService(config)


def quick_clone_voice(text: str, reference_audio: str,
                     output_filename: Optional[str] = None,
                     enable_vllm: bool = False) -> VoiceGenerationResult:
    """
    快速语音克隆

    Args:
        text: 要生成的文本
        reference_audio: 参考音频路径
        output_filename: 输出文件名（可选）
        enable_vllm: 是否启用VLLM加速

    Returns:
        VoiceGenerationResult: 生成结果
    """
    config = ServiceConfig(enable_vllm=enable_vllm)
    service = get_voice_service(config)

    return service.clone_voice(
        text=text,
        reference_audio=reference_audio,
        output_filename=output_filename
    )


def clone_voice_with_vllm(text: str, reference_audio: str,
                         output_filename: Optional[str] = None) -> VoiceGenerationResult:
    """
    使用VLLM加速的语音克隆

    Args:
        text: 要生成的文本
        reference_audio: 参考音频路径
        output_filename: 输出文件名（可选）

    Returns:
        VoiceGenerationResult: 生成结果
    """
    return quick_clone_voice(text, reference_audio, output_filename, enable_vllm=True)


# ==================== 示例和测试代码 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    def test_basic_functionality():
        """测试基本功能"""
        print("=" * 60)
        print("CosyVoice语音生成服务测试（简化版）")
        print("=" * 60)

        try:
            # 创建服务实例
            config = ServiceConfig(
                enable_vllm=True,
                log_level="INFO"
            )

            with get_voice_service(config) as service:
                # 测试服务状态
                status = service.get_service_status()
                print(f"服务状态: {status['service']['initialized']}")
                print(f"VLLM启用: {status['service']['config']['enable_vllm']}")
                print(f"CosyVoice可用: {status['cosyvoice_available']}")

                if status['cosyvoice_available']:
                    print("\n✅ CosyVoice服务可用，开始测试...")

                    # 这里需要实际的音频文件来测试
                    print("\n📝 测试语音克隆（需要参考音频文件）:")
                    print("result = service.clone_voice(")
                    print('    text="你好，这是测试语音。",')
                    print('    reference_audio="path/to/reference.wav"')
                    print(")")
                    print("if result.is_success:")
                    print("    print(f'克隆成功: {result.audio_path}')")
                    print("else:")
                    print("    print(f'克隆失败: {result.error_message}')")

                else:
                    print("\n❌ CosyVoice服务不可用，请检查模型和依赖")

        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

    def test_convenience_functions():
        """测试便捷函数"""
        print("\n" + "=" * 60)
        print("便捷函数测试")
        print("=" * 60)

        try:
            # 测试便捷函数导入
            from voice_generator import quick_clone_voice, clone_voice_with_vllm

            print("📦 便捷函数导入成功")

            print("\n📝 快速克隆示例:")
            print("result = quick_clone_voice('你好世界', 'reference.wav')")

            print("\n📝 VLLM加速克隆示例:")
            print("result = clone_voice_with_vllm('你好世界', 'reference.wav')")

            print("✅ 便捷函数测试通过")

        except Exception as e:
            print(f"❌ 便捷函数测试失败: {e}")

    # 运行测试
    test_basic_functionality()
    test_convenience_functions()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n📖 使用示例:")
    print("from backend.voice_generator import get_voice_service, quick_clone_voice")
    print()
    print("# 简单克隆")
    print("result = quick_clone_voice('你好世界', 'reference.wav')")
    print()
    print("# VLLM加速克隆")
    print("result = clone_voice_with_vllm('你好世界', 'reference.wav')")
    print()
    print("# 使用服务实例")
    print("service = get_voice_service()")
    print("result = service.clone_voice('测试文本', 'ref.wav')")
    print("if result.is_success:")
    print("    print(f'生成成功: {result.audio_path}')")