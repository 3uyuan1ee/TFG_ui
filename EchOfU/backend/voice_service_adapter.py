"""
语音服务适配器 - 现有代码无缝升级到UnifiedVoiceService

使用方法：
1. 替换原来的 OpenVoiceService.get() 为 VoiceServiceAdapter.get()
2. 所有现有接口保持不变，底层自动选择最佳引擎
"""

from typing import Optional, List, Dict, Any
from .path_manager import PathManager
from .unified_voice_service import UnifiedVoiceService, VoiceEngine


class VoiceServiceAdapter:
    """
    语音服务适配器 - 保持与现有OpenVoiceService接口的兼容性
    同时提供CosyVoice3的高级功能
    """

    _instance = None
    _lock = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            import threading
            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not VoiceServiceAdapter._initialized:
            self._initialize_service()
            VoiceServiceAdapter._initialized = True

    def _initialize_service(self):
        """初始化统一语音服务"""
        print("[VoiceServiceAdapter] ===============================================")
        print("[VoiceServiceAdapter] 初始化统一语音服务...")
        print("[VoiceServiceAdapter] ===============================================")

        # 初始化PathManager
        self.path_manager = PathManager()

        # 初始化统一语音服务
        self.unified_service = UnifiedVoiceService(self.path_manager)

        # 为了兼容性，保留一些属性
        self.feature_manager = self.unified_service.feature_manager

        print("[VoiceServiceAdapter] 统一语音服务初始化完成")
        status = self.unified_service.get_service_status()
        print(f"[VoiceServiceAdapter] OpenVoice可用: {status['openvoice_available']}")
        print(f"[VoiceServiceAdapter] CosyVoice3可用: {status['cosyvoice3_available']}")
        print(f"[VoiceServiceAdapter] 总说话人数: {status['total_speakers']}")
        print("==============================================")

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get(cls):
        """简化的获取方法"""
        return cls.get_instance()

    def list_available_speakers(self) -> List[str]:
        """列出可用的说话人ID（兼容性方法）"""
        speakers = self.unified_service.list_available_speakers()
        return [speaker['id'] for speaker in speakers]

    def extract_and_save_speaker_feature(self,
                                       speaker_id: str,
                                       reference_audio: str,
                                       preferred_engine: str = "cosyvoice3") -> bool:
        """
        提取并保存说话人特征（增强版本）

        Args:
            speaker_id: 说话人编号
            reference_audio: 参考音频路径
            preferred_engine: 首选引擎 ("openvoice" 或 "cosyvoice3")

        Returns:
            是否提取成功
        """
        return self.unified_service.extract_speaker_feature(
            speaker_id, reference_audio, preferred_engine
        )

    def generate_speech(self,
                       text: str,
                       speaker_id: Optional[str] = None,
                       engine: Optional[str] = None,
                       **kwargs) -> Optional[str]:
        """
        生成语音（增强版本）

        Args:
            text: 目标文本
            speaker_id: 说话人ID
            engine: 指定引擎
            **kwargs: 其他参数（emotion, instruction, speed等）

        Returns:
            生成的音频文件路径
        """
        if speaker_id is None:
            # 基础TTS模式，使用OpenVoice
            return self.unified_service.openvoice_service.generate_speech(text)
        else:
            # 使用统一服务
            return self.unified_service.generate_speech(
                text=text,
                speaker_id=speaker_id,
                engine=engine,
                **kwargs
            )

    # 新增的高级功能方法
    def generate_speech_with_emotion(self,
                                   text: str,
                                   speaker_id: str,
                                   emotion: str,
                                   engine: Optional[str] = None) -> Optional[str]:
        """
        使用情感控制生成语音

        Args:
            text: 目标文本
            speaker_id: 说话人ID
            emotion: 情感 (sad, happy, angry, excited, calm等)
            engine: 指定引擎

        Returns:
            生成的音频文件路径
        """
        return self.unified_service.generate_speech(
            text=text,
            speaker_id=speaker_id,
            engine=engine,
            emotion=emotion
        )

    def generate_speech_with_instruction(self,
                                        text: str,
                                        speaker_id: str,
                                        instruction: str,
                                        engine: Optional[str] = None) -> Optional[str]:
        """
        使用指令控制生成语音（CosyVoice3的强大功能）

        Args:
            text: 目标文本
            speaker_id: 说话人ID
            instruction: 语音指令（如"请用广东话表达"、"请用悲伤的语气"等）
            engine: 指定引擎

        Returns:
            生成的音频文件路径
        """
        return self.unified_service.generate_speech(
            text=text,
            speaker_id=speaker_id,
            engine=engine,
            instruction=instruction
        )

    def get_available_engines(self) -> List[str]:
        """获取可用的语音引擎"""
        return self.unified_service.get_available_engines()

    def get_engine_recommendations(self, text: str, **kwargs) -> Dict[str, Any]:
        """获取引擎推荐"""
        return self.unified_service.get_engine_recommendations(text, **kwargs)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return self.unified_service.get_service_status()

    def get_speaker_details(self) -> List[Dict[str, Any]]:
        """获取说话人详细信息（包含使用的引擎）"""
        return self.unified_service.list_available_speakers()

    def delete_speaker(self, speaker_id: str) -> bool:
        """删除说话人特征"""
        return self.unified_service.delete_speaker_feature(speaker_id)

    # 高级功能示例方法
    def generate_multilingual_speech(self,
                                    text: str,
                                    speaker_id: str,
                                    target_language: str = "zh") -> Optional[str]:
        """
        生成多语言语音（CosyVoice3功能）

        Args:
            text: 目标文本
            speaker_id: 说话人ID
            target_language: 目标语言 (zh, en, jp, ko, yue等)

        Returns:
            生成的音频文件路径
        """
        return self.unified_service.generate_speech(
            text=text,
            speaker_id=speaker_id,
            engine="cosyvoice3",
            target_language=target_language
        )

    def batch_generate_speech(self,
                             texts: List[str],
                             speaker_id: str,
                             engine: Optional[str] = None,
                             **kwargs) -> List[Optional[str]]:
        """
        批量生成语音

        Args:
            texts: 文本列表
            speaker_id: 说话人ID
            engine: 指定引擎
            **kwargs: 其他参数

        Returns:
            生成的音频文件路径列表
        """
        results = []
        for text in texts:
            result = self.generate_speech(
                text=text,
                speaker_id=speaker_id,
                engine=engine,
                **kwargs
            )
            results.append(result)
        return results

    # 兼容性方法 - 委托给OpenVoiceService
    @property
    def speaker_features(self):
        """获取说话人特征（兼容性属性）"""
        return self.unified_service.speaker_features

    @property
    def device(self):
        """获取设备类型（兼容性属性）"""
        return self.unified_service.openvoice_service.device

    def load_speaker_features(self):
        """加载说话人特征（兼容性方法）"""
        return self.unified_service.openvoice_service.load_speaker_features()

    def check_models_exist(self):
        """检查模型是否存在（兼容性方法）"""
        return self.unified_service.openvoice_service.check_models_exist()

    def download_openvoice_models(self):
        """下载OpenVoice模型（兼容性方法）"""
        return self.unified_service.openvoice_service.download_openvoice_models()


# 使用示例和说明
"""
使用说明：

1. 替换现有代码：
   原来：OpenVoiceService.get()
   现在：VoiceServiceAdapter.get()

2. 基础功能（完全兼容）：
   - extract_and_save_speaker_feature(speaker_id, audio_path)
   - generate_speech(text, speaker_id)
   - list_available_speakers()

3. 新增高级功能：
   - generate_speech_with_emotion(text, speaker_id, "happy")
   - generate_speech_with_instruction(text, speaker_id, "请用广东话表达")
   - generate_multilingual_speech(text, speaker_id, "en")
   - batch_generate_speech(texts, speaker_id)

4. 引擎选择：
   - get_available_engines() 查看可用引擎
   - get_engine_recommendations() 获取推荐
   - 在方法中指定 engine="cosyvoice3" 使用CosyVoice3

5. 服务状态：
   - get_service_status() 查看详细状态
   - get_speaker_details() 查看说话人详情
   - delete_speaker() 删除说话人
"""