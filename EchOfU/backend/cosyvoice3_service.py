import os
import sys
import time
import torch
import torchaudio
import threading
from typing import Optional, Dict, Any, Generator
from pathlib import Path

# CosyVoice3导入
AutoModel = None
cosyvoice_path = None

def init_cosyvoice_import():
    """延迟初始化CosyVoice3导入"""
    global AutoModel, cosyvoice_path
    if AutoModel is not None:
        return True

    try:
        from .path_manager import PathManager
        path_manager = PathManager()
        cosyvoice_path = path_manager.get_cosyvoice_path()

        if os.path.exists(cosyvoice_path):
            sys.path.insert(0, cosyvoice_path)
            try:
                from cosyvoice.cli.cosyvoice import AutoModel
                print("[CosyVoice3Service] CosyVoice3导入成功")
                return True
            except ImportError as e:
                # 处理常见的依赖问题
                if "wetext" in str(e):
                    print("[CosyVoice3Service] 缺少wetext依赖，请安装: pip install wetext")
                elif "ttsfrd" in str(e):
                    print("[CosyVoice3Service] 缺少ttsfrd依赖，请安装: pip install ttsfrd")
                else:
                    print(f"[CosyVoice3Service] CosyVoice3导入失败: {e}")
                return False
            except Exception as e:
                print(f"[CosyVoice3Service] CosyVoice3导入异常: {e}")
                return False
        else:
            print(f"[CosyVoice3Service] CosyVoice3目录不存在: {cosyvoice_path}")
            return False

    except Exception as e:
        print(f"[CosyVoice3Service] 初始化失败: {e}")
        return False

from .path_manager import PathManager


class CosyVoice3Service:
    """
    CosyVoice3服务 - 高级语音克隆和合成
    特点：
    1. 支持零样本语音克隆
    2. 多语言和方言支持
    3. 精细的语音控制（情感、语速、呼吸等）
    4. 说话人特征缓存机制
    """

    def __init__(self, path_manager: PathManager, model_dir: str = None):
        self.path_manager = path_manager
        self.model = None
        self.model_dir = model_dir or self._get_default_model_dir()
        self.sample_rate = 22050
        self._speaker_cache = {}  # 说话人特征缓存
        self._lock = threading.Lock()

        # 初始化CosyVoice3导入
        init_cosyvoice_import()

    def _get_default_model_dir(self) -> str:
        """获取默认模型路径"""
        return self.path_manager.get_root_begin_path(
            "CosyVoice",
            "pretrained_models/Fun-CosyVoice3-0.5B"
        )

    def _lazy_load_model(self):
        """延迟加载模型"""
        if AutoModel is None:
            raise RuntimeError("CosyVoice3未正确安装或导入失败")

        if self.model is None:
            with self._lock:
                if self.model is None:
                    print(f"[CosyVoice3Service] 正在加载模型: {self.model_dir}")
                    start_time = time.time()

                    try:
                        # 检查模型目录是否存在
                        if not os.path.exists(self.model_dir):
                            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

                        self.model = AutoModel(model_dir=self.model_dir)
                        self.sample_rate = self.model.sample_rate

                        load_time = time.time() - start_time
                        print(f"[CosyVoice3Service] 模型加载完成，耗时: {load_time:.2f}秒")

                    except Exception as e:
                        print(f"[CosyVoice3Service] 模型加载失败: {e}")
                        raise

    def clone_voice_zero_shot(self,
                            target_text: str,
                            reference_audio: str,
                            instruction: str = "You are a helpful assistant.<|endofprompt|>",
                            speed: float = 1.0) -> Optional[str]:
        """
        零样本语音克隆

        Args:
            target_text: 目标文本
            reference_audio: 参考音频路径
            instruction: 提示指令
            speed: 语速控制

        Returns:
            生成的音频文件路径
        """
        self._lazy_load_model()

        try:
            print(f"[CosyVoice3Service] 开始零样本克隆")
            print(f"  目标文本: {target_text[:50]}...")
            print(f"  参考音频: {reference_audio}")

            # 确保参考音频存在
            if not os.path.exists(reference_audio):
                raise FileNotFoundError(f"参考音频不存在: {reference_audio}")

            # 构建完整的提示文本
            if instruction and not instruction.endswith('<|endofprompt|>'):
                instruction += '<|endofprompt|>'

            # 生成输出路径
            timestamp = int(time.time() * 1000)
            output_path = self.path_manager.get_temp_voice_path(
                f"cosyvoice3_zs_{timestamp}.wav"
            )

            # 执行零样本推理
            audio_results = []
            for i, result in enumerate(self.model.inference_zero_shot(
                tts_text=target_text,
                prompt_text=instruction,
                prompt_wav=reference_audio,
                stream=False,
                speed=speed
            )):
                audio_results.append(result['tts_speech'])
                print(f"  生成音频片段 {i+1}: {result['tts_speech'].shape}")

            if audio_results:
                # 合并所有音频片段
                final_audio = torch.cat(audio_results, dim=1)
                torchaudio.save(output_path, final_audio, self.sample_rate)

                duration = final_audio.shape[1] / self.sample_rate
                print(f"[CosyVoice3Service] 零样本克隆完成，时长: {duration:.2f}秒")
                print(f"  输出文件: {output_path}")

                return output_path
            else:
                print("[CosyVoice3Service] 警告: 未生成任何音频")
                return None

        except Exception as e:
            print(f"[CosyVoice3Service] 零样本克隆失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clone_voice_cross_lingual(self,
                                 target_text: str,
                                 reference_audio: str,
                                 target_language: str = 'zh',
                                 speed: float = 1.0) -> Optional[str]:
        """
        跨语言语音克隆

        Args:
            target_text: 目标文本
            reference_audio: 参考音频路径
            target_language: 目标语言 (zh, en, jp, ko, yue等)
            speed: 语速控制

        Returns:
            生成的音频文件路径
        """
        self._lazy_load_model()

        try:
            print(f"[CosyVoice3Service] 开始跨语言克隆")
            print(f"  目标语言: {target_language}")
            print(f"  目标文本: {target_text[:50]}...")

            # 语言标记映射
            lang_tokens = {
                'zh': '<|zh|>',
                'en': '<|en|>',
                'jp': '<|jp|>',
                'ko': '<|ko|>',
                'yue': '<|yue|>',  # 粤语
                'de': '<|de|>',
                'es': '<|es|>',
                'fr': '<|fr|>',
                'it': '<|it|>',
                'ru': '<|ru|>'
            }

            lang_token = lang_tokens.get(target_language, '<|zh|>')
            text_with_lang = f"{lang_token}{target_text}"

            # 生成输出路径
            timestamp = int(time.time() * 1000)
            output_path = self.path_manager.get_temp_voice_path(
                f"cosyvoice3_cl_{target_language}_{timestamp}.wav"
            )

            # 执行跨语言推理
            audio_results = []
            for result in self.model.inference_cross_lingual(
                tts_text=text_with_lang,
                prompt_wav=reference_audio,
                stream=False,
                speed=speed
            ):
                audio_results.append(result['tts_speech'])

            if audio_results:
                final_audio = torch.cat(audio_results, dim=1)
                torchaudio.save(output_path, final_audio, self.sample_rate)

                duration = final_audio.shape[1] / self.sample_rate
                print(f"[CosyVoice3Service] 跨语言克隆完成，时长: {duration:.2f}秒")
                return output_path
            else:
                return None

        except Exception as e:
            print(f"[CosyVoice3Service] 跨语言克隆失败: {e}")
            return None

    def instruct_voice_synthesis(self,
                                target_text: str,
                                instruction: str,
                                reference_audio: Optional[str] = None,
                                cached_speaker_id: Optional[str] = None,
                                speed: float = 1.0) -> Optional[str]:
        """
        指令控制语音合成（CosyVoice3最强大的功能）

        Args:
            target_text: 目标文本
            instruction: 语音指令（如"请用广东话表达"、"请用悲伤的语气"等）
            reference_audio: 可选的参考音频
            cached_speaker_id: 可选的缓存说话人ID
            speed: 语速控制

        Returns:
            生成的音频文件路径
        """
        self._lazy_load_model()

        try:
            print(f"[CosyVoice3Service] 开始指令控制合成")
            print(f"  指令: {instruction}")
            print(f"  目标文本: {target_text[:50]}...")

            # 构建完整指令
            if not instruction.endswith('<|endofprompt|>'):
                instruction += '<|endofprompt|>'

            # 生成输出路径
            timestamp = int(time.time() * 1000)
            output_path = self.path_manager.get_temp_voice_path(
                f"cosyvoice3_instruct_{timestamp}.wav"
            )

            # 确定使用的参考音频或说话人ID
            prompt_wav = reference_audio if reference_audio and os.path.exists(reference_audio) else None
            zero_shot_spk_id = cached_speaker_id if cached_speaker_id in self._speaker_cache else None

            # 执行指令推理
            audio_results = []
            for result in self.model.inference_instruct2(
                tts_text=target_text,
                instruct_text=instruction,
                prompt_wav=prompt_wav,
                zero_shot_spk_id=zero_shot_spk_id or '',
                stream=False,
                speed=speed
            ):
                audio_results.append(result['tts_speech'])

            if audio_results:
                final_audio = torch.cat(audio_results, dim=1)
                torchaudio.save(output_path, final_audio, self.sample_rate)

                duration = final_audio.shape[1] / self.sample_rate
                print(f"[CosyVoice3Service] 指令控制合成完成，时长: {duration:.2f}秒")
                return output_path
            else:
                return None

        except Exception as e:
            print(f"[CosyVoice3Service] 指令控制合成失败: {e}")
            return None

    def cache_speaker(self,
                     speaker_id: str,
                     reference_audio: str,
                     description: str = "") -> bool:
        """
        缓存说话人特征，供后续使用

        Args:
            speaker_id: 说话人ID
            reference_audio: 参考音频路径
            description: 说话人描述

        Returns:
            是否缓存成功
        """
        self._lazy_load_model()

        try:
            print(f"[CosyVoice3Service] 缓存说话人特征: {speaker_id}")

            # 构建提示文本
            prompt_text = description or "You are a helpful assistant."
            if not prompt_text.endswith('<|endofprompt|>'):
                prompt_text += '<|endofprompt|>'

            # 缓存说话人
            success = self.model.add_zero_shot_spk(
                prompt_text=prompt_text,
                prompt_wav=reference_audio,
                spk_id=speaker_id
            )

            if success:
                self._speaker_cache[speaker_id] = {
                    'reference_audio': reference_audio,
                    'description': description,
                    'created_at': time.time()
                }
                print(f"[CosyVoice3Service] 说话人 {speaker_id} 缓存成功")
                return True
            else:
                print(f"[CosyVoice3Service] 说话人 {speaker_id} 缓存失败")
                return False

        except Exception as e:
            print(f"[CosyVoice3Service] 缓存说话人失败: {e}")
            return False

    def get_cached_speakers(self) -> Dict[str, Any]:
        """获取已缓存的说话人列表"""
        return self._speaker_cache.copy()

    def remove_cached_speaker(self, speaker_id: str) -> bool:
        """移除缓存的说话人"""
        if speaker_id in self._speaker_cache:
            del self._speaker_cache[speaker_id]
            print(f"[CosyVoice3Service] 已移除缓存说话人: {speaker_id}")
            return True
        return False

    def synthesize_with_fine_control(self,
                                    target_text: str,
                                    emotion: str = None,
                                    speed: float = 1.0,
                                    volume: float = 1.0,
                                    pitch: str = None,
                                    reference_audio: str = None) -> Optional[str]:
        """
        精细控制的语音合成

        Args:
            target_text: 目标文本
            emotion: 情感控制 (sad, happy, angry, excited等)
            speed: 语速控制
            volume: 音量控制
            pitch: 音调控制
            reference_audio: 参考音频

        Returns:
            生成的音频文件路径
        """
        # 构建精细控制指令
        control_parts = []

        if emotion:
            emotion_map = {
                'sad': '请用悲伤的语气',
                'happy': '请用开心的语气',
                'angry': '请用愤怒的语气',
                'excited': '请用兴奋的语气',
                'calm': '请用平静的语气',
                'whisper': '请用耳语的方式',
                'shouting': '请用大声喊的方式'
            }
            if emotion in emotion_map:
                control_parts.append(emotion_map[emotion])

        if speed != 1.0:
            if speed > 1.0:
                control_parts.append(f'请用{speed:.1f}倍的语速')
            else:
                control_parts.append(f'请用慢一些的语速')

        if volume != 1.0:
            if volume > 1.0:
                control_parts.append('请用大一些的音量')
            else:
                control_parts.append('请用小一些的音量')

        if pitch:
            pitch_map = {
                'high': '请用高一些的音调',
                'low': '请用低一些的音调',
                'normal': '请用正常的音调'
            }
            if pitch in pitch_map:
                control_parts.append(pitch_map[pitch])

        # 组合指令
        instruction = ' '.join(control_parts) + '。<|endofprompt|>'

        return self.instruct_voice_synthesis(
            target_text=target_text,
            instruction=instruction,
            reference_audio=reference_audio
        )

    def is_available(self) -> bool:
        """检查CosyVoice3是否可用"""
        try:
            if AutoModel is None:
                return False

            model_path = Path(self.model_dir)
            if not model_path.exists():
                return False

            # 检查关键文件
            required_files = ['cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt']
            for file in required_files:
                if not (model_path / file).exists():
                    return False

            return True
        except Exception:
            return False