import os
import time
import json
import subprocess
import torch
from datetime import datetime
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor


class PathManager:
    """路径管理器，统一处理所有路径相关逻辑"""

    def __init__(self):
        self.project_root = self._get_project_root()

    def _get_project_root(self):
        """获取项目根目录路径 - 递归向上查找EchOfU目录"""
        def find_echofu_root(start_dir):
            current_dir = os.path.abspath(start_dir)

            # 如果当前目录名是EchOfU，检查是否有OpenVoice目录
            if os.path.basename(current_dir) == "EchOfU":
                if os.path.exists(os.path.join(current_dir, "OpenVoice")):
                    return current_dir

            # 如果已经到达根目录还没找到，返回None
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                return None

            # 递归向上查找
            return find_echofu_root(parent_dir)

        result = find_echofu_root(os.getcwd())
        if result is None:
            raise FileNotFoundError("无法找到EchOfU项目根目录")
        return result

    def get_model_path(self, *path_parts):
        """获取模型相关路径"""
        return os.path.join(self.project_root, *path_parts)

    def get_openvoice_v2_path(self, *path_parts):
        """获取OpenVoice V2相关路径"""
        return self.get_model_path("OpenVoice/checkpoints_v2", *path_parts)

    def get_speaker_features_path(self):
        """获取说话人特征文件路径"""
        return os.path.join(self.project_root, "models/OpenVoice/speaker_features.json")

    def get_output_voice_path(self, timestamp):
        """生成输出语音文件路径"""
        return os.path.join(self.project_root, f"static/voices/generated_{timestamp}.wav")


class ModelDownloader:
    """模型下载器，处理模型文件的下载和解压"""

    @staticmethod
    def download_with_progress(url, output_path):
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

    @staticmethod
    def extract_zip_file(zip_path, extract_dir):
        """解压zip文件"""
        import zipfile

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"[OpenVoice] 解压完成: {zip_path} -> {extract_dir}")

        except Exception as e:
            print(f"[OpenVoice] 解压失败: {e}")
            raise


class AudioProcessor:
    """音频处理器，处理音频相关的操作"""

    def __init__(self):
        self.path_manager = PathManager()

    def extract_audio_from_video(self, video_path):
        """从视频中提取音频"""
        try:
            # 确保输出目录存在 - 使用PathManager
            audio_dir = self.path_manager.get_model_path("static/audios")
            os.makedirs(audio_dir, exist_ok=True)

            # 生成音频文件名 - 使用PathManager
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_output = os.path.join(audio_dir, f"extracted_{timestamp}.wav")

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


class SpeakerFeatureManager:
    """说话人特征管理器 - 专门处理说话人特征的加载、保存和管理"""

    def __init__(self, path_manager):
        self.path_manager = path_manager
        self.speaker_features = {}
        self._load_all_features()

    def _load_all_features(self):
        """加载所有已保存的说话人特征"""
        features_file = self.path_manager.get_speaker_features_path()
        if os.path.exists(features_file):
            try:
                with open(features_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 加载特征张量
                for speaker_id, info in data.items():
                    feature_path = info['feature_path']
                    if os.path.exists(feature_path):
                        self.speaker_features[speaker_id] = {
                            'se': torch.load(feature_path, map_location='cpu'),
                            'reference_audio': info['reference_audio'],
                            'created_time': info['created_time']
                        }

                print(f"[SpeakerFeatureManager] 已加载 {len(self.speaker_features)} 个说话人特征")
            except Exception as e:
                print(f"[SpeakerFeatureManager] 加载说话人特征失败: {e}")

    def save_feature(self, speaker_id, reference_audio, se_tensor):
        """保存说话人特征"""
        try:
            # 保存特征张量 - 使用PathManager获取绝对路径
            feature_path = self.path_manager.get_model_path("models/OpenVoice", f"{speaker_id}_se.pth")
            torch.save(se_tensor, feature_path)

            # 保存元数据
            metadata = {
                'feature_path': feature_path,
                'reference_audio': reference_audio,
                'created_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # 更新特征文件
            features_file = self.path_manager.get_speaker_features_path()
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

            print(f"[SpeakerFeatureManager] 说话人特征已保存: {speaker_id}")

        except Exception as e:
            print(f"[SpeakerFeatureManager] 保存说话人特征失败: {e}")

    def get_feature(self, speaker_id):
        """获取指定说话人的特征"""
        return self.speaker_features.get(speaker_id)

    def list_speakers(self):
        """列出所有可用的说话人"""
        return list(self.speaker_features.keys())

    def extract_feature(self, speaker_id, reference_audio, tone_converter):
        """提取说话人特征"""
        try:
            if not tone_converter:
                print(f"[SpeakerFeatureManager] 转换器未初始化")
                return False

            print(f"[SpeakerFeatureManager] 开始提取说话人特征: {speaker_id}")

            # 提取说话人特征 - 使用PathManager获取processed目录的绝对路径
            processed_dir = self.path_manager.get_model_path("processed")
            target_se_result = se_extractor.get_se(
                reference_audio,
                vc_model=tone_converter,
                target_dir=processed_dir
            )

            # get_se返回元组(se_tensor, audio_name) 只需要张量部分
            if isinstance(target_se_result, tuple):
                target_se = target_se_result[0]
            else:
                target_se = target_se_result

            # 保存特征
            self.save_feature(speaker_id, reference_audio, target_se)

            print(f"[SpeakerFeatureManager] 说话人特征提取完成: {speaker_id}")
            return True

        except Exception as e:
            print(f"[SpeakerFeatureManager] 说话人特征提取失败: {e}")
            return False


class ModelManager:
    """模型管理器 - 专门处理模型的初始化、下载和管理"""

    def __init__(self, path_manager):
        self.path_manager = path_manager
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tone_converter = None

    def initialize(self):
        """初始化模型"""
        try:
            # 确保目录存在
            self._ensure_directories()

            # 检查模型是否存在
            if not self._check_models_exist():
                print("[ModelManager] 检测到模型文件缺失，开始自动下载...")
                self._download_models()

            # 配置模型路径 - 使用PathManager
            config_path = self.path_manager.get_openvoice_v2_path("converter/config.json")
            base_ckpt = self.path_manager.get_openvoice_v2_path("converter/checkpoint.pth")

            # 加载音色转换器
            self.tone_converter = ToneColorConverter(config_path, device=self.device)
            self.tone_converter.load_ckpt(base_ckpt)

            print(f"[ModelManager] V2模型加载完成，使用设备: {self.device}")
            return True

        except Exception as e:
            print(f"[ModelManager] 模型初始化失败: {e}")
            return False

    def _ensure_directories(self):
        """确保必要目录存在"""
        dirs = [
            self.path_manager.get_openvoice_v2_path(),
            self.path_manager.get_model_path("OpenVoice/checkpoints/base_speakers"),
            self.path_manager.get_model_path("models/OpenVoice"),
            self.path_manager.get_model_path("static/voices"),
            self.path_manager.get_model_path("processed")
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _check_models_exist(self):
        """检查模型文件是否存在"""
        required_files = [
            self.path_manager.get_openvoice_v2_path("converter/config.json"),
            self.path_manager.get_openvoice_v2_path("converter/checkpoint.pth")
        ]

        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(f"[ModelManager] 缺失模型文件: {missing}")
            return False
        return True

    def _download_models(self):
        """下载OpenVoice V2预训练模型"""
        print("[ModelManager] 下载OpenVoice V2预训练模型...")

        try:
            # V2模型下载地址
            zip_url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

            # 使用PathManager获取路径
            project_root = self.path_manager.project_root
            zip_path = os.path.join(project_root, "OpenVoice/checkpoints_v2_0417.zip")
            extract_dir = os.path.join(project_root, "OpenVoice/")

            print(f"[ModelManager] 下载V2检查点压缩包...")
            ModelDownloader.download_with_progress(zip_url, zip_path)

            # 解压压缩包
            print(f"[ModelManager] 解压检查点文件...")
            ModelDownloader.extract_zip_file(zip_path, extract_dir)

            # 清理压缩包文件
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"[ModelManager] 清理压缩包文件: {zip_path}")

            print("[ModelManager] V2模型下载和解压完成")

        except Exception as e:
            print(f"[ModelManager] 模型下载失败: {e}")
            raise


class VoiceGenerator:
    """语音生成器 - 专门处理语音合成功能"""

    def __init__(self, path_manager):
        self.path_manager = path_manager

    def generate_with_melotts_tts(self, text, output_path, base_speaker_key="ZH"):
        """尝试使用MeloTTS生成语音"""
        try:
            import os

            print(f"[VoiceGenerator] 开始使用MeloTTS生成语音...")
            print(f"[VoiceGenerator] 输入文本: {text}")
            print(f"[VoiceGenerator] 输出路径: {output_path}")
            print(f"[VoiceGenerator] 说话人key: {base_speaker_key}")

            # 设置环境变量，强制使用CPU，避免MPS问题
            print(f"[VoiceGenerator] 设置环境变量强制使用CPU...")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print(f"[VoiceGenerator] PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
            print(f"[VoiceGenerator] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

            print(f"[VoiceGenerator] 正在导入MeloTTS...")
            from melo.api import TTS
            print(f"[VoiceGenerator] MeloTTS导入成功")

            # 根据base_speaker_key确定语言
            language_mapping = {
                "EN_NEWEST": "EN", "EN": "EN",
                "ES": "ES", "FR": "FR",
                "ZH": "ZH", "JP": "JP", "KR": "KR"
            }

            language = language_mapping.get(base_speaker_key, "EN")
            print(f"[VoiceGenerator] 映射后语言: {language}")

            print(f"[VoiceGenerator] 正在初始化MeloTTS模型 (语言: {language}, 设备: cpu)...")
            model = TTS(language=language, device='cpu')
            print(f"[VoiceGenerator] MeloTTS模型初始化成功")

            speaker_ids = model.hps.data.spk2id
            print(f"[VoiceGenerator] 可用说话人: {dict(speaker_ids)}")

            # 选择说话人ID
            if base_speaker_key in speaker_ids:
                speaker_id = speaker_ids[base_speaker_key]
                print(f"[VoiceGenerator] 使用指定说话人: {base_speaker_key} -> {speaker_id}")
            else:
                speaker_id = list(speaker_ids.values())[0]
                print(f"[VoiceGenerator] 未找到说话人 {base_speaker_key}，使用默认说话人: {speaker_id}")

            # 生成语音
            speed = 1.0
            print(f"[VoiceGenerator] 开始生成语音 (语速: {speed})...")
            print(f"[VoiceGenerator] 模型参数: 语言={language}, 说话人ID={speaker_id}, 输出路径={output_path}")

            model.tts_to_file(text, speaker_id, output_path, speed=speed)

            # 检查文件是否成功生成
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"[VoiceGenerator] MeloTTS生成语音成功: {output_path}")
                print(f"[VoiceGenerator] 生成文件大小: {file_size} bytes")
                return True
            else:
                print(f"[VoiceGenerator] MeloTTS生成失败: 输出文件不存在 {output_path}")
                return False

        except ImportError as ie:
            print(f"[VoiceGenerator] MeloTTS导入错误: {ie}")
            print("[VoiceGenerator] MeloTTS未安装")
            return False
        except Exception as e:
            print(f"[VoiceGenerator] MeloTTS生成失败 - 详细错误: {type(e).__name__}: {e}")
            print(f"[VoiceGenerator] 错误详情: {str(e)}")
            import traceback
            print(f"[VoiceGenerator] 错误堆栈:")
            traceback.print_exc()
            return False


class OpenVoiceService:
    """OpenVoice服务主类 - 协调各个组件"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenVoiceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            print("[OpenVoice] ===============================================")
            print("[OpenVoice] 初始化OpenVoice服务...")
            print("[OpenVoice] ===============================================")

            # 初始化组件
            print("[OpenVoice] 初始化PathManager...")
            self.path_manager = PathManager()
            print(f"[OpenVoice] PathManager初始化完成，项目根目录: {self.path_manager.project_root}")

            print("[OpenVoice] 初始化ModelManager...")
            self.model_manager = ModelManager(self.path_manager)

            print("[OpenVoice] 初始化SpeakerFeatureManager...")
            self.feature_manager = SpeakerFeatureManager(self.path_manager)

            print("[OpenVoice] 初始化VoiceGenerator...")
            self.voice_generator = VoiceGenerator(self.path_manager)

            print("[OpenVoice] 初始化AudioProcessor...")
            self.audio_processor = AudioProcessor()

            # 初始化模型
            print("[OpenVoice] 开始初始化模型组件...")
            # 确保tts_model属性存在（V2版本主要使用MeloTTS）
            self.tts_model = None
            self._initialize_components()
            OpenVoiceService._initialized = True

            print("[OpenVoice] ===============================================")
            print("[OpenVoice] OpenVoice服务初始化完成")
            print(f"[OpenVoice] 服务ID: {id(self)}")
            print(f"[OpenVoice] 音色转换器状态: {'✅ 已加载' if self.tone_converter else '❌ 未加载'}")
            print(f"[OpenVoice] 已加载说话人数量: {len(self.feature_manager.speaker_features)}")
            print("==============================================")

    def _initialize_components(self):
        """初始化所有组件"""
        print("[OpenVoice] 开始组件初始化流程...")
        if self.model_manager.initialize():
            self.tone_converter = self.model_manager.tone_converter
            print("[OpenVoice] ✅ 组件初始化完成")
            print(f"[OpenVoice] 设备类型: {self.model_manager.device}")
            print(f"[OpenVoice] 音色转换器: {type(self.tone_converter).__name__}")
        else:
            print("[OpenVoice] ❌ 模型管理器初始化失败")
            self._fallback_to_default_state()

    def _fallback_to_default_state(self):
        """组件初始化失败时的默认状态"""
        print("[OpenVoice] 进入默认状态（功能受限）")
        self.tone_converter = None
        self.tts_model = None  # 确保属性存在
        print("[OpenVoice] ⚠️  使用默认状态，语音功能不可用")

    @property
    def device(self):
        """获取设备类型"""
        return self.model_manager.device

    @property
    def speaker_features(self):
        """获取说话人特征"""
        return self.feature_manager.speaker_features

    def list_available_speakers(self):
        """列出可用的说话人"""
        return self.feature_manager.list_speakers()

    def extract_and_save_speaker_feature(self, speaker_id, reference_audio):
        """提取并保存说话人特征（public : 供外部函数调用）"""
        return self.feature_manager.extract_feature(speaker_id, reference_audio, self.tone_converter)

    def generate_speech(self, text, speaker_id=None):
        """生成语音 - 统一接口（V2版本）"""
        print("[OpenVoice] ===============================================")
        print("[OpenVoice] 开始语音生成流程...")
        print(f"[OpenVoice] 输入文本: '{text}'")
        print(f"[OpenVoice] 说话人ID: {speaker_id if speaker_id else 'None (使用基础TTS)'}")

        try:
            if not self.tone_converter:
                print("[OpenVoice] ❌ 音色转换器未初始化，无法生成语音")
                return None

            print("[OpenVoice] ✅ 音色转换器已就绪")

            # 生成输出文件名 - 使用PathManager
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.path_manager.get_output_voice_path(timestamp)
            print(f"[OpenVoice] 输出文件路径: {output_path}")

            if speaker_id and speaker_id in self.feature_manager.speaker_features:
                # 使用说话人克隆（V2方式）
                print(f"[OpenVoice] 🎭 使用说话人克隆模式: {speaker_id}")
                print(f"[OpenVoice] 可用说话人: {list(self.feature_manager.speaker_features.keys())}")
                return self._clone_voice_with_cached_feature(text, speaker_id, output_path)
            else:
                # 使用基础TTS（MeloTTS）
                print("[OpenVoice] 🎤 使用基础TTS模式（MeloTTS）")
                if speaker_id:
                    print(f"[OpenVoice] ⚠️  说话人 {speaker_id} 不存在，切换到基础TTS")
                return self._generate_base_speech(text, output_path)

        except Exception as e:
            print(f"[OpenVoice] ❌ 语音生成异常: {type(e).__name__}: {e}")
            import traceback
            print(f"[OpenVoice] 错误堆栈:")
            traceback.print_exc()
            return None

    def _clone_voice_with_cached_feature(self, text, speaker_id, output_path, base_speaker_key="ZH"):
        """使用缓存的说话人特征进行语音克隆"""
        try:
            if not self.tone_converter:
                return None

            speaker_info = self.feature_manager.get_feature(speaker_id)
            target_se = speaker_info['se']

            # 1. 先用基础模型生成语音（使用MeloTTS）
            temp_audio = os.path.join(self.path_manager.project_root, f"temp_base_{speaker_id}_{int(time.time())}.wav")
            base_speaker_path = self._generate_base_speech(text, temp_audio, base_speaker_key)

            if not base_speaker_path:
                return None

            # 2. 加载基础说话人特征 - 使用PathManager
            source_se_path = self.path_manager.get_model_path("OpenVoice/checkpoints_v2/base_speakers/ses", f"{base_speaker_key.lower()}.pth")
            if os.path.exists(source_se_path):
                source_se = torch.load(source_se_path, map_location=self.model_manager.device)
            else:
                print(f"[OpenVoice] 未找到基础说话人特征: {source_se_path}")
                return None

            # 3. 使用缓存的特征进行音色转换
            encode_message = "@MyShell"
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

    def _generate_base_speech(self, text, output_path, base_speaker_key="ZH"):
        """生成基础语音（支持MeloTTS）"""
        try:
            # 尝试使用MeloTTS
            if self.voice_generator.generate_with_melotts_tts(text, output_path, base_speaker_key):
                return output_path
            else:
                print("[OpenVoice] MeloTTS生成失败")
                return None

        except Exception as e:
            print(f"[OpenVoice] 基础语音生成失败: {e}")
            return None

    # 兼容性方法 - 委托给相应的组件，确保不产生递归调用
    def initialize_model(self):
        """初始化模型（兼容性方法）"""
        return self._initialize_components()

    def load_speaker_features(self):
        """加载已保存的说话人特征（兼容性方法）"""
        return self.feature_manager.speaker_features

    def save_speaker_feature(self, speaker_id, reference_audio, se_tensor):
        """保存说话人特征（兼容性方法）"""
        return self.feature_manager.save_feature(speaker_id, reference_audio, se_tensor)

    def clone_voice_with_cached_feature(self, text, speaker_id, output_path, base_speaker_key="ZH"):
        """使用缓存的说话人特征进行语音克隆（兼容性方法）"""
        return self._clone_voice_with_cached_feature(text, speaker_id, output_path, base_speaker_key)

    def generate_base_speech(self, text, output_path, base_speaker_key="ZH"):
        """生成基础语音（兼容性方法）"""
        return self._generate_base_speech(text, output_path, base_speaker_key)

    def try_melotts_tts(self, text, output_path, base_speaker_key="ZH"):
        """尝试使用MeloTTS生成语音（兼容性方法）"""
        return self.voice_generator.generate_with_melotts_tts(text, output_path, base_speaker_key)

    def synthesize_speech(self, text, output_path, speaker="default", language="Chinese"):
        """快速语音合成 - 兼容性方法"""
        try:
            # 使用MeloTTS作为基础生成
            return self._generate_base_speech(text, output_path, "ZH")

        except Exception as e:
            print(f"[OpenVoice] 语音合成失败: {e}")
            return None

    def ensure_directories(self):
        """确保必要目录存在（兼容性方法）"""
        return self.model_manager._ensure_directories()

    def check_models_exist(self):
        """检查模型文件是否存在（兼容性方法）"""
        return self.model_manager._check_models_exist()

    def download_openvoice_models(self):
        """下载OpenVoice V2预训练模型（兼容性方法）"""
        return self.model_manager._download_models()


def generate_voice(text, speaker_id=None):
    """
    统一的语音生成接口（V2版本）

    Args:
        text (str): 要合成的文本
        speaker_id (str, optional): 说话人ID，如果为None则使用基础TTS

    Returns:
        str: 生成的语音文件路径，失败返回None
    """
    print("[OpenVoice] ===============================================")
    print("[OpenVoice] 外部接口 - 语音生成请求")
    print(f"[OpenVoice] 输入参数: 文本='{text}', 说话人ID={speaker_id}")
    print("==============================================")

    try:
        # 获取OpenVoice服务实例
        print("[OpenVoice] 获取OpenVoice服务实例...")
        openvoice_service = OpenVoiceService()
        print(f"[OpenVoice] 服务实例ID: {id(openvoice_service)}")

        # 检查模型是否可用（主要检查音色转换器）
        if not openvoice_service.tone_converter:
            print("[OpenVoice] ❌ 音色转换器未正确加载，请检查模型文件")
            return None

        print("[OpenVoice] ✅ 音色转换器已就绪，开始生成语音")

        # 生成语音（V2架构）
        print("[OpenVoice] 调用内部语音生成接口...")
        result_path = openvoice_service.generate_speech(text, speaker_id)

        if result_path and os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"[OpenVoice] ✅ 外部接口 - 语音生成成功")
            print(f"[OpenVoice] 输出文件: {result_path}")
            print(f"[OpenVoice] 文件大小: {file_size} bytes")
            print("==============================================")
            return result_path
        else:
            print("[OpenVoice] ❌ 外部接口 - 语音生成失败")
            print("==============================================")
            return None

    except Exception as e:
        print(f"[OpenVoice] ❌ 外部接口 - 语音生成异常: {type(e).__name__}: {e}")
        import traceback
        print(f"[OpenVoice] 错误详情:")
        traceback.print_exc()
        print("==============================================")
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
    """
    提取说话人特征(符合前端设计的接口）

    Args:
        data (dict): 包含ref_video和speaker_id的字典

    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        video_path = data['ref_video']
        print("[backend.model_trainer] 开始OpenVoice语音特征提取...")

        extracted_audio = None
        if video_path.endswith(".mp4"):
            # 从视频中提取音频，使用AudioProcessor
            audio_processor = AudioProcessor()
            extracted_audio = audio_processor.extract_audio_from_video(video_path)

        if not extracted_audio:
            print("[backend.model_trainer] 音频提取失败，无法进行特征提取")
            return False

        # ToDo : 可以让用户自定义语音名字 （需要在前端加上speaker_id选项)

        # 获取或生成说话人ID
        speaker_id = data.get("speaker_id")
        if not speaker_id:
            # 生成的说话人ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            speaker_id = f"user_{timestamp}"

        # 使用重构后的接口提取并保存说话人特征
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

            # 使用PathManager获取正确的项目根目录路径
            path_manager = PathManager()
            info_path = path_manager.get_model_path("models/OpenVoice", f"{speaker_id}_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"[backend.model_trainer] 模型信息已保存: {info_path}")
            return True

        else:
            print("[backend.model_trainer] OpenVoice说话人特征提取失败")
            return False

    except Exception as e:
        print(f"[backend.model_trainer] OpenVoice训练失败: {e}")
        return False