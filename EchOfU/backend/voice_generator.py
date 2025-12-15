import os
import time
import json
from datetime import datetime
import subprocess
import json
import torch
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor

class OpenVoiceService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenVoiceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.initialize_model()
            OpenVoiceService._initialized = True

    def initialize_model(self):
        """初始化模型"""
        try:
            # 确保目录存在
            self.ensure_directories()

            # 检查模型是否存在
            if not self.check_models_exist():
                print("[OpenVoice] 检测到模型文件缺失，开始自动下载...")
                self.download_openvoice_models()

            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # 获取项目根目录路径
            current_dir = os.getcwd()
            if current_dir.endswith("/EchOfU"):
                # 如果当前在EchOfU目录下，使用相对路径
                project_root = "."
            else:
                # 如果在其他目录，使用绝对路径
                project_root = "EchOfU"

            # 配置路径 - 使用V2版本
            config_path = os.path.join(project_root, "OpenVoice/checkpoints_v2/converter/config.json")
            base_ckpt = os.path.join(project_root, "OpenVoice/checkpoints_v2/converter/checkpoint.pth")  # V2版本使用checkpoint.pth

            # 加载音色转换器（V2中converter.pth是主要模型文件）
            self.tone_converter = ToneColorConverter(config_path, device=self.device)
            self.tone_converter.load_ckpt(base_ckpt)

            # 初始化TTS模型为None，将使用MeloTTS
            self.tts_model = None  # V2版本推荐使用MeloTTS

            # 加载已保存的说话人特征
            self.speaker_features = self.load_speaker_features()

            print(f"[OpenVoice] V2模型加载完成，使用设备: {self.device}")
            print(f"[OpenVoice] 已加载 {len(self.speaker_features)} 个说话人特征")
            print("[OpenVoice] 注意：V2版本推荐使用MeloTTS作为基础TTS引擎")

        except Exception as e:
            print(f"[OpenVoice] 模型初始化失败: {e}")
            # 如果模型文件不存在，返回默认状态
            self.fallback_to_default_state()

    def fallback_to_default_state(self):
        """模型加载失败时的默认状态"""
        self.tts_model = None
        self.tone_converter = None
        self.speaker_features = {}
        print("[OpenVoice] 使用默认状态，语音功能不可用")

    def load_speaker_features(self):
        """加载已保存的说话人特征"""
        features_file = "models/OpenVoice/speaker_features.json"
        if os.path.exists(features_file):
            try:
                with open(features_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 加载特征张量
                features = {}
                for speaker_id, info in data.items():
                    feature_path = info['feature_path']
                    if os.path.exists(feature_path):
                        features[speaker_id] = {
                            'se': torch.load(feature_path, map_location='cpu'),
                            'reference_audio': info['reference_audio'],
                            'created_time': info['created_time']
                        }

                return features
            except Exception as e:
                print(f"[OpenVoice] 加载说话人特征失败: {e}")
                return {}
        return {}

    def save_speaker_feature(self, speaker_id, reference_audio, se_tensor):
        """保存说话人特征"""
        try:
            # 保存特征张量
            feature_path = f"models/OpenVoice/{speaker_id}_se.pth"
            torch.save(se_tensor, feature_path)

            # 保存元数据
            metadata = {
                'feature_path': feature_path,
                'reference_audio': reference_audio,
                'created_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # 更新特征文件
            features_file = "models/OpenVoice/speaker_features.json"
            all_features = {}
            if os.path.exists(features_file):
                with open(features_file, 'r', encoding='utf-8') as f:
                    all_features = json.load(f)

            all_features[speaker_id] = metadata

            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(all_features, f, indent=2, ensure_ascii=False)

            # 更新内存中的特征
            self.speaker_features[speaker_id] = {
                'se': se_tensor,
                'reference_audio': reference_audio,
                'created_time': metadata['created_time']
            }

            print(f"[OpenVoice] 说话人特征已保存: {speaker_id}")

        except Exception as e:
            print(f"[OpenVoice] 保存说话人特征失败: {e}")

    def extract_and_save_speaker_feature(self, speaker_id, reference_audio):
        """提取并保存说话人特征（public : 供外部函数调用）"""
        try:
            if not self.tone_converter:
                print(f"[OpenVoice] 转换器未初始化")
                return False

            print(f"[OpenVoice] 开始提取说话人特征: {speaker_id}")

            # 提取说话人特征
            target_se_result = se_extractor.get_se(
                reference_audio,
                vc_model=self.tone_converter,
                target_dir="processed"
            )

            # get_se返回元组(se_tensor, audio_name) 只需要张量部分
            if isinstance(target_se_result, tuple):
                target_se = target_se_result[0]
            else:
                target_se = target_se_result

            # 保存特征
            self.save_speaker_feature(speaker_id, reference_audio, target_se)

            print(f"[OpenVoice] 说话人特征提取完成: {speaker_id}")
            return True

        except Exception as e:
            print(f"[OpenVoice] 说话人特征提取失败: {e}")
            return False

    def list_available_speakers(self):
        """列出可用的说话人"""
        return list(self.speaker_features.keys())

    def generate_speech(self, text, speaker_id=None):
        """生成语音 - 统一接口（V2版本）"""
        try:
            if not self.tone_converter:
                print("[OpenVoice] 音色转换器未初始化")
                return None

            # 生成输出文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"static/voices/generated_{timestamp}.wav"

            if speaker_id and speaker_id in self.speaker_features:
                # 使用说话人克隆（V2方式）
                return self.clone_voice_with_cached_feature(text, speaker_id, output_path)
            else:
                # 使用基础TTS（MeloTTS或传统TTS）
                return self.generate_base_speech(text, output_path)

        except Exception as e:
            print(f"[OpenVoice] 语音生成失败: {e}")
            return None

    def clone_voice_with_cached_feature(self, text, speaker_id, output_path, base_speaker_key="ZH"):
        """使用缓存的说话人特征进行语音克隆"""
        try:
            if not self.tone_converter:
                return None

            speaker_info = self.speaker_features[speaker_id]
            target_se = speaker_info['se']

            # 1. 先用基础模型生成语音（使用MeloTTS或base speaker）
            temp_audio = f"temp_base_{speaker_id}_{int(time.time())}.wav"
            base_speaker_path = self.generate_base_speech(text, temp_audio, base_speaker_key)

            if not base_speaker_path:
                return None

            # 2. 加载基础说话人特征
            # 获取项目根目录路径
            current_dir = os.getcwd()
            if current_dir.endswith("/EchOfU"):
                project_root = "."
            else:
                project_root = "EchOfU"

            source_se_path = os.path.join(project_root, "OpenVoice/checkpoints_v2/base_speakers/ses", f"{base_speaker_key.lower()}.pth")
            if os.path.exists(source_se_path):
                source_se = torch.load(source_se_path, map_location=self.device)
            else:
                # 如果没有预训练的基础说话人特征，使用默认特征
                print(f"[OpenVoice] 未找到基础说话人特征: {source_se_path}")
                return None

            # 3. 使用缓存的特征进行音色转换
            encode_message = "@MyShell"  # V2版本的编码消息
            self.tone_converter.convert(
                audio_src_path=temp_audio,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
                message=encode_message
            )

            # 4. 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            print(f"[OpenVoice] 使用V2模型和缓存特征生成语音: {speaker_id}")
            return output_path

        except Exception as e:
            print(f"[OpenVoice] 使用缓存特征生成语音失败: {e}")
            return None

    def generate_base_speech(self, text, output_path, base_speaker_key="ZH"):
        """生成基础语音（支持MeloTTS和传统TTS）"""
        try:
            # 尝试使用MeloTTS
            if self.try_melotts_tts(text, output_path, base_speaker_key):
                return output_path

            # 回退到传统TTS
            if self.tts_model:
                self.tts_model.tts(
                    text=text,
                    output_path=output_path,
                    speaker="default",
                    language="Chinese",
                    speed=1.0
                )
                return output_path
            else:
                print("[OpenVoice] 没有可用的TTS引擎")
                return None

        except Exception as e:
            print(f"[OpenVoice] 基础语音生成失败: {e}")
            return None

    def try_melotts_tts(self, text, output_path, base_speaker_key="ZH"):
        """尝试使用MeloTTS生成语音"""
        try:
            import os

            # ToDo : Mac上改为了CPU，记得改回来
            # 设置环境变量，强制使用CPU，避免MPS问题
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            from melo.api import TTS

            # 根据base_speaker_key确定语言
            language_mapping = {
                "EN_NEWEST": "EN", "EN": "EN",
                "ES": "ES", "FR": "FR",
                "ZH": "ZH", "JP": "JP", "KR": "KR"
            }

            language = language_mapping.get(base_speaker_key, "EN")

            # 强制使用CPU初始化MeloTTS，避免MPS相关错误
            model = TTS(language=language, device='cpu')
            speaker_ids = model.hps.data.spk2id

            # 选择说话人ID
            if base_speaker_key in speaker_ids:
                speaker_id = speaker_ids[base_speaker_key]
            else:
                # 使用默认说话人
                speaker_id = list(speaker_ids.values())[0]
                print(f"[OpenVoice] 未找到说话人 {base_speaker_key}，使用默认说话人")

            # ToDo : 可以让用户自己调节语速
            # 生成语音
            speed = 1.0
            model.tts_to_file(text, speaker_id, output_path, speed=speed)

            print(f"[OpenVoice] MeloTTS生成语音成功: {output_path}")
            return True

        except ImportError:
            print("[OpenVoice] MeloTTS未安装，回退到传统TTS")
            return False
        except Exception as e:
            print(f"[OpenVoice] MeloTTS生成失败: {e}")
            return False

    def synthesize_speech(self, text, output_path, speaker="default", language="Chinese"):
        """快速语音合成 - 兼容性方法"""
        try:
            # 如果有TTS模型，使用传统方法
            if self.tts_model:
                self.tts_model.tts(
                    text=text,
                    output_path=output_path,
                    speaker=speaker,
                    language=language,
                    speed=1.0
                )
                return output_path
            else:
                # 使用MeloTTS作为基础生成
                return self.generate_base_speech(text, output_path, "ZH")

        except Exception as e:
            print(f"[OpenVoice] 语音合成失败: {e}")
            return None

    def ensure_directories(self):
        """确保必要目录存在"""
        # 获取项目根目录路径
        current_dir = os.getcwd()
        if current_dir.endswith("/EchOfU"):
            project_root = "."
        else:
            project_root = "EchOfU"

        dirs = [
            os.path.join(project_root, "OpenVoice/checkpoints_v2"),
            os.path.join(project_root, "OpenVoice/checkpoints/base_speakers"),
            "models/OpenVoice",
            "static/voices",
            "processed"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def check_models_exist(self):
        """检查模型文件是否存在"""
        # 获取项目根目录路径
        current_dir = os.getcwd()
        if current_dir.endswith("/EchOfU"):
            project_root = "."
        else:
            project_root = "EchOfU"

        required_files = [
            os.path.join(project_root, "OpenVoice/checkpoints_v2/converter/config.json"),
            os.path.join(project_root, "OpenVoice/checkpoints_v2/converter/checkpoint.pth")
        ]

        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(f"[OpenVoice] 缺失模型文件: {missing}")
            return False
        return True

    def download_openvoice_models(self):
        """下载OpenVoice V2预训练模型"""
        print("[OpenVoice] 下载OpenVoice V2预训练模型...")

        try:
            # V2模型下载地址
            zip_url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

            # 获取当前工作目录和项目根目录
            current_dir = os.getcwd()
            if current_dir.endswith("/EchOfU"):
                # 如果当前在EchOfU目录下，使用相对路径
                project_root = "."
            else:
                # 如果在其他目录，使用绝对路径
                project_root = "EchOfU"

            zip_path = os.path.join(project_root, "OpenVoice/checkpoints_v2_0417.zip")
            extract_dir = os.path.join(project_root, "OpenVoice/")

            print(f"[OpenVoice] 下载V2检查点压缩包...")
            self.download_with_progress(zip_url, zip_path)

            # 解压压缩包
            print(f"[OpenVoice] 解压检查点文件...")
            self.extract_zip_file(zip_path, extract_dir)

            # 清理压缩包文件
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"[OpenVoice] 清理压缩包文件: {zip_path}")

            print("[OpenVoice] V2模型下载和解压完成")

        except Exception as e:
            print(f"[OpenVoice] 模型下载失败: {e}")
            raise

    def download_with_progress(self, url, output_path):
        """文件下载进度显示"""
        import requests

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r[OpenVoice] 下载进度: {progress:.1f}%", end='', flush=True)

            print()  # 换行

        except Exception as e:
            print(f"[OpenVoice] 下载失败: {e}")
            raise

    def extract_zip_file(self, zip_path, extract_dir):
        """解压zip文件"""
        try:
            import zipfile

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"[OpenVoice] 解压完成: {zip_path} -> {extract_dir}")

        except Exception as e:
            print(f"[OpenVoice] 解压失败: {e}")
            raise

def generate_voice(text, speaker_id=None):
    """
    统一的语音生成接口（V2版本）

    Args:
        text (str): 要合成的文本
        speaker_id (str, optional): 说话人ID，如果为None则使用基础TTS

    Returns:
        str: 生成的语音文件路径，失败返回None
    """
    try:
        # 获取OpenVoice服务实例
        openvoice_service = OpenVoiceService()

        # 检查模型是否可用（主要检查音色转换器）
        if not openvoice_service.tone_converter:
            print("[OpenVoice] 音色转换器未正确加载，请检查模型文件")
            return None

        # 生成语音（V2架构）
        result_path = openvoice_service.generate_speech(text, speaker_id)

        if result_path and os.path.exists(result_path):
            print(f"[OpenVoice] 语音生成成功: {result_path}")
            return result_path
        else:
            print("[OpenVoice] 语音生成失败")
            return None

    except Exception as e:
        print(f"[OpenVoice] 语音生成异常: {e}")
        return None

def extract_speaker_feature(speaker_id, reference_audio):
    """
    提取说话人特征

    Args:
        speaker_id (str): 说话人唯一标识
        reference_audio (str): 参考音频文件路径

    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        openvoice_service = OpenVoiceService()
        return openvoice_service.extract_and_save_speaker_feature(speaker_id, reference_audio)
    except Exception as e:
        print(f"[OpenVoice] 提取说话人特征异常: {e}")
        return False

def list_available_speakers():
    """
    列出所有可用的说话人

    Returns:
        list: 说话人ID列表
    """
    try:
        openvoice_service = OpenVoiceService()
        return openvoice_service.list_available_speakers()
    except Exception as e:
        print(f"[OpenVoice] 列出说话人失败: {e}")
        return []

def extract_trait_from_audio(data):
    try:
        """
        提取说话人特征(最终接口）
    
        Args:
            speaker_id (str): 说话人唯一标识
            reference_audio (str): 参考音频文件路径
    
        Returns:
            bool: 成功返回True，失败返回False
        """
        video_path = data['ref_video']

        print("[backend.model_trainer] 开始OpenVoice语音特征提取...")

        if video_path.endswith(".mp4"):
            # 从视频中提取音频
            extracted_audio = extract_audio_from_video(video_path)

        # ToDo : 可以让用户自定义语音名字 （需要在前端加上speaker_id选项)
        speaker_id=data["speaker_id"]
        if speaker_id:
            print()
        else:
            # 生成的说话人ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            speaker_id = f"user_{timestamp}"

        # 提取并保存说话人特征
        success = extract_speaker_feature(speaker_id, extracted_audio)

        if success:
            print(f"[backend.model_trainer] OpenVoice说话人特征提取成功: {speaker_id}")

            # 保存训练信息
            model_info = {
                "model_type": "OpenVoice",
                "speaker_id": speaker_id,
                "reference_video": video_path,
                "extracted_audio": extracted_audio,
                "training_time": datetime.now().isoformat(),
                "status": "completed"
            }

            info_path = f"models/OpenVoice/{speaker_id}_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"[backend.model_trainer] 模型信息已保存: {info_path}")

        else:
            print("[backend.model_trainer] OpenVoice说话人特征提取失败")

    except Exception as e:
        print(f"[backend.model_trainer] OpenVoice训练失败: {e}")


def extract_audio_from_video(video_path):
    """
    从视频中提取音频
    """
    try:
        # 确保输出目录存在
        os.makedirs("static/audios", exist_ok=True)

        # 生成音频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_output = f"static/audios/extracted_{timestamp}.wav"

        # 使用ffmpeg提取音频
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # 禁用视频
            "-acodec", "pcm_s16le",  # 音频编码
            "-ar", "16000",  # 采样率
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            audio_output
        ]

        print(f"[backend.model_trainer] 提取音频命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1分钟超时
        )

        if result.returncode == 0:
            print(f"[backend.model_trainer] 音频提取成功: {audio_output}")
            return audio_output
        else:
            print(f"[backend.model_trainer] 音频提取失败: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("[backend.model_trainer] 音频提取超时")
        return None
    except Exception as e:
        print(f"[backend.model_trainer] 音频提取异常: {e}")
        return None
