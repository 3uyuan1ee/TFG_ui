import os
import json
import time
import threading
from typing import Dict, Any, Optional, List
from enum import Enum

from .path_manager import PathManager
from .voice_generator import SpeakerFeatureManager, OpenVoiceService
from .cosyvoice3_service import CosyVoice3Service


class VoiceEngine(Enum):
    """语音引擎类型"""
    OPENVOICE = "openvoice"
    COSYVOICE3 = "cosyvoice3"


class UnifiedVoiceService:
    """
    统一语音服务 - 无缝集成OpenVoice和CosyVoice3
    保持现有的speaker_id接口，底层可以选择使用不同的语音引擎
    """

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.openvoice_service = None
        self.cosyvoice3_service = None
        self.feature_manager = None  # 将在_init_services中初始化
        self._lock = threading.Lock()

        # 特征存储结构：兼容两种引擎
        # {
        #     'speaker_id': {
        #         'engine': 'openvoice' | 'cosyvoice3',
        #         'feature': OpenVoice特征 | CosyVoice3说话人ID,
        #         'reference_audio': '原始音频路径',
        #         'created_time': '创建时间',
        #         'metadata': {...}
        #     }
        # }
        self.speaker_features = {}

        # 加载已保存的特征
        self._load_features()

        # 初始化服务
        self._init_services()

    def _init_services(self):
        """初始化语音服务"""
        try:
            print("[UnifiedVoiceService] 初始化语音服务...")

            # 初始化OpenVoice（使用正确的单例模式方法）
            self.openvoice_service = OpenVoiceService.get_instance()
            print("[UnifiedVoiceService] OpenVoice服务初始化成功")

            # 初始化SpeakerFeatureManager（适配器需要）
            self.feature_manager = self.openvoice_service.feature_manager

            # 初始化CosyVoice3（如果可用）
            try:
                self.cosyvoice3_service = CosyVoice3Service(self.path_manager)
                if self.cosyvoice3_service.is_available():
                    print("[UnifiedVoiceService] CosyVoice3服务初始化成功")
                else:
                    print("[UnifiedVoiceService] CosyVoice3不可用，仅使用OpenVoice")
                    self.cosyvoice3_service = None
            except Exception as cosyvoice_error:
                print(f"[UnifiedVoiceService] CosyVoice3初始化失败: {cosyvoice_error}")
                self.cosyvoice3_service = None

        except Exception as e:
            print(f"[UnifiedVoiceService] 服务初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_features(self):
        """加载已保存的说话人特征"""
        try:
            features_file = self.path_manager.get_speaker_features_path()
            if os.path.exists(features_file):
                with open(features_file, 'r', encoding='utf-8') as f:
                    self.speaker_features = json.load(f)
                print(f"[UnifiedVoiceService] 加载了 {len(self.speaker_features)} 个说话人特征")
        except Exception as e:
            print(f"[UnifiedVoiceService] 加载特征失败: {e}")
            self.speaker_features = {}

    def _save_features(self):
        """保存说话人特征"""
        try:
            features_file = self.path_manager.get_speaker_features_path()
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(self.speaker_features, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[UnifiedVoiceService] 保存特征失败: {e}")

    def get_available_engines(self) -> List[str]:
        """获取可用的语音引擎"""
        engines = [VoiceEngine.OPENVOICE.value]
        if self.cosyvoice3_service and self.cosyvoice3_service.is_available():
            engines.append(VoiceEngine.COSYVOICE3.value)
        return engines

    def extract_speaker_feature(self,
                               speaker_id: str,
                               reference_audio: str,
                               preferred_engine: str = VoiceEngine.OPENVOICE.value) -> bool:
        """
        提取说话人特征

        Args:
            speaker_id: 说话人编号
            reference_audio: 参考音频路径
            preferred_engine: 首选引擎

        Returns:
            是否提取成功
        """
        with self._lock:
            try:
                print(f"[UnifiedVoiceService] 开始提取说话人特征: {speaker_id}")
                print(f"  参考音频: {reference_audio}")
                print(f"  首选引擎: {preferred_engine}")

                # 检查参考音频是否存在
                if not os.path.exists(reference_audio):
                    print(f"[UnifiedVoiceService] 参考音频不存在: {reference_audio}")
                    return False

                # 确定使用的引擎
                engine = preferred_engine
                if engine == VoiceEngine.COSYVOICE3.value:
                    if not self.cosyvoice3_service or not self.cosyvoice3_service.is_available():
                        print("[UnifiedVoiceService] CosyVoice3不可用，回退到OpenVoice")
                        engine = VoiceEngine.OPENVOICE.value
                else:
                    engine = VoiceEngine.OPENVOICE.value

                # 提取特征
                feature_data = None
                success = False

                if engine == VoiceEngine.COSYVOICE3.value:
                    # 使用CosyVoice3提取特征
                    success = self.cosyvoice3_service.cache_speaker(
                        speaker_id=speaker_id,
                        reference_audio=reference_audio,
                        description=f"Speaker {speaker_id}"
                    )

                    if success:
                        feature_data = {
                            'engine': VoiceEngine.COSYVOICE3.value,
                            'cached_speaker_id': speaker_id,
                            'reference_audio': reference_audio,
                            'created_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'metadata': {
                                'extraction_method': 'cosyvoice3_cache',
                                'model_version': 'Fun-CosyVoice3-0.5B'
                            }
                        }

                else:
                    # 使用OpenVoice提取特征
                    success = self.feature_manager.extract_feature(
                        speaker_id, reference_audio, self.openvoice_service.tone_converter
                    )

                    if success:
                        feature = self.feature_manager.get_feature(speaker_id)
                        if feature:
                            feature_data = {
                                'engine': VoiceEngine.OPENVOICE.value,
                                'feature_path': f"processed/{speaker_id}_se.pth",  # 相对路径
                                'reference_audio': reference_audio,
                                'created_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'metadata': {
                                    'extraction_method': 'openvoice_se_extractor',
                                    'model_version': 'OpenVoiceV2'
                                }
                            }

                if success and feature_data:
                    # 保存特征信息
                    self.speaker_features[speaker_id] = feature_data
                    self._save_features()

                    print(f"[UnifiedVoiceService] 特征提取成功: {speaker_id} (引擎: {engine})")
                    return True
                else:
                    print(f"[UnifiedVoiceService] 特征提取失败: {speaker_id}")
                    return False

            except Exception as e:
                print(f"[UnifiedVoiceService] 特征提取异常: {e}")
                import traceback
                traceback.print_exc()
                return False

    def list_available_speakers(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的说话人

        Returns:
            说话人列表，包含ID、引擎、创建时间等信息
        """
        speakers = []
        for speaker_id, feature_data in self.speaker_features.items():
            speakers.append({
                'id': speaker_id,
                'engine': feature_data['engine'],
                'reference_audio': feature_data['reference_audio'],
                'created_time': feature_data['created_time'],
                'metadata': feature_data.get('metadata', {})
            })

        return speakers

    def generate_speech(self,
                       text: str,
                       speaker_id: str,
                       engine: str = None,
                       **kwargs) -> Optional[str]:
        """
        生成语音

        Args:
            text: 目标文本
            speaker_id: 说话人编号
            engine: 指定引擎，如果为None则使用该说话人对应的引擎
            **kwargs: 其他参数（emotion, speed, instruction等）

        Returns:
            生成的音频文件路径
        """
        try:
            print(f"[UnifiedVoiceService] 开始语音生成")
            print(f"  文本: {text[:50]}...")
            print(f"  说话人: {speaker_id}")

            # 检查说话人是否存在
            if speaker_id not in self.speaker_features:
                print(f"[UnifiedVoiceService] 说话人不存在: {speaker_id}")
                return None

            feature_data = self.speaker_features[speaker_id]

            # 确定使用的引擎
            if engine is None:
                engine = feature_data['engine']

            # 验证引擎可用性
            if engine == VoiceEngine.COSYVOICE3.value:
                if not self.cosyvoice3_service or not self.cosyvoice3_service.is_available():
                    print("[UnifiedVoiceService] CosyVoice3不可用，回退到OpenVoice")
                    engine = VoiceEngine.OPENVOICE.value
            else:
                engine = VoiceEngine.OPENVOICE.value

            # 根据引擎生成语音
            if engine == VoiceEngine.COSYVOICE3.value:
                return self._generate_with_cosyvoice3(text, speaker_id, feature_data, **kwargs)
            else:
                return self._generate_with_openvoice(text, speaker_id, feature_data, **kwargs)

        except Exception as e:
            print(f"[UnifiedVoiceService] 语音生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_with_cosyvoice3(self,
                                 text: str,
                                 speaker_id: str,
                                 feature_data: Dict[str, Any],
                                 **kwargs) -> Optional[str]:
        """使用CosyVoice3生成语音"""
        try:
            print("[UnifiedVoiceService] 使用CosyVoice3生成语音")

            # 提取参数
            emotion = kwargs.get('emotion')
            speed = kwargs.get('speed', 1.0)
            instruction = kwargs.get('instruction')

            # 如果有自定义指令，使用指令合成
            if instruction:
                return self.cosyvoice3_service.instruct_voice_synthesis(
                    target_text=text,
                    instruction=instruction,
                    cached_speaker_id=speaker_id,
                    speed=speed
                )

            # 如果有情感控制，使用精细控制
            elif emotion or speed != 1.0:
                return self.cosyvoice3_service.synthesize_with_fine_control(
                    target_text=text,
                    emotion=emotion,
                    speed=speed,
                    reference_audio=feature_data['reference_audio']
                )

            # 否则使用零样本克隆
            else:
                return self.cosyvoice3_service.clone_voice_zero_shot(
                    target_text=text,
                    reference_audio=feature_data['reference_audio'],
                    speed=speed
                )

        except Exception as e:
            print(f"[UnifiedVoiceService] CosyVoice3生成失败: {e}")
            # 回退到OpenVoice
            print("[UnifiedVoiceService] 回退到OpenVoice")
            return self._generate_with_openvoice(text, speaker_id, feature_data, **kwargs)

    def _generate_with_openvoice(self,
                                text: str,
                                speaker_id: str,
                                feature_data: Dict[str, Any],
                                **kwargs) -> Optional[str]:
        """使用OpenVoice生成语音"""
        try:
            print("[UnifiedVoiceService] 使用OpenVoice生成语音")

            # 获取OpenVoice特征
            openvoice_feature = self.feature_manager.get_feature(speaker_id)
            if not openvoice_feature:
                print(f"[UnifiedVoiceService] 无法获取OpenVoice特征: {speaker_id}")
                return None

            # 使用OpenVoice生成语音
            return self.openvoice_service.generate_speech_with_feature(
                text=text,
                target_se=openvoice_feature['se'],
                **kwargs
            )

        except Exception as e:
            print(f"[UnifiedVoiceService] OpenVoice生成失败: {e}")
            return None

    def delete_speaker_feature(self, speaker_id: str) -> bool:
        """删除说话人特征"""
        with self._lock:
            try:
                if speaker_id not in self.speaker_features:
                    return False

                feature_data = self.speaker_features[speaker_id]

                # 根据引擎删除特征
                if feature_data['engine'] == VoiceEngine.COSYVOICE3.value:
                    if self.cosyvoice3_service:
                        self.cosyvoice3_service.remove_cached_speaker(speaker_id)

                # 从特征列表中删除
                del self.speaker_features[speaker_id]
                self._save_features()

                # 如果是OpenVoice，也要从SpeakerFeatureManager中删除
                if feature_data['engine'] == VoiceEngine.OPENVOICE.value:
                    self.feature_manager.speaker_features.pop(speaker_id, None)

                print(f"[UnifiedVoiceService] 已删除说话人特征: {speaker_id}")
                return True

            except Exception as e:
                print(f"[UnifiedVoiceService] 删除说话人特征失败: {e}")
                return False

    def get_engine_recommendations(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        根据输入参数推荐最适合的引擎

        Returns:
            推荐结果，包含推荐引擎和原因
        """
        recommendations = {
            'recommended_engine': VoiceEngine.OPENVOICE.value,
            'confidence': 0.5,
            'reason': '默认使用OpenVoice',
            'alternatives': []
        }

        # 检查特殊需求
        has_emotion_control = 'emotion' in kwargs
        has_instruction = 'instruction' in kwargs
        has_cross_lingual = 'target_language' in kwargs
        text_length = len(text)

        # 如果需要情感控制或指令，推荐CosyVoice3
        if (has_emotion_control or has_instruction) and self.cosyvoice3_service:
            recommendations['recommended_engine'] = VoiceEngine.COSYVOICE3.value
            recommendations['confidence'] = 0.9
            recommendations['reason'] = '需要情感或指令控制，CosyVoice3更擅长'

        # 如果需要跨语言，推荐CosyVoice3
        elif has_cross_lingual and self.cosyvoice3_service:
            recommendations['recommended_engine'] = VoiceEngine.COSYVOICE3.value
            recommendations['confidence'] = 0.85
            recommendations['reason'] = '需要跨语言支持，CosyVoice3更优秀'

        # 如果文本很长，推荐CosyVoice3（更好的流式处理）
        elif text_length > 200 and self.cosyvoice3_service:
            recommendations['recommended_engine'] = VoiceEngine.COSYVOICE3.value
            recommendations['confidence'] = 0.7
            recommendations['reason'] = '长文本处理，CosyVoice3效率更高'

        # 添加备选方案
        if self.cosyvoice3_service and recommendations['recommended_engine'] == VoiceEngine.OPENVOICE.value:
            recommendations['alternatives'].append({
                'engine': VoiceEngine.COSYVOICE3.value,
                'reason': '如果需要高级功能，可以尝试CosyVoice3'
            })

        if recommendations['recommended_engine'] == VoiceEngine.COSYVOICE3.value:
            recommendations['alternatives'].append({
                'engine': VoiceEngine.OPENVOICE.value,
                'reason': '如果追求稳定性，可以使用OpenVoice'
            })

        return recommendations

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'openvoice_available': self.openvoice_service is not None,
            'cosyvoice3_available': self.cosyvoice3_service is not None and self.cosyvoice3_service.is_available(),
            'total_speakers': len(self.speaker_features),
            'openvoice_speakers': len([s for s in self.speaker_features.values() if s['engine'] == VoiceEngine.OPENVOICE.value]),
            'cosyvoice3_speakers': len([s for s in self.speaker_features.values() if s['engine'] == VoiceEngine.COSYVOICE3.value]),
            'available_engines': self.get_available_engines()
        }