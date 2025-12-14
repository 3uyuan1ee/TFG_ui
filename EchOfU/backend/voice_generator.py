import os
import time
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
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 配置路径 - 使用V2版本
            config_path = "EchOfU/OpenVoice/checkpoints_v2/config.json"
            base_ckpt = "EchOfU/OpenVoice/checkpoints_v2/checkpoints.pth"
            converter_ckpt = "EchOfU/OpenVoice/checkpoints_v2/converter.pth"

            # 加载基础TTS模型
            self.tts_model = BaseSpeakerTTS(config_path, device=device)
            self.tts_model.load_ckpt(base_ckpt)

            # 加载音色转换器
            self.tone_converter = ToneColorConverter(config_path, device=device)
            self.tone_converter.load_ckpt(converter_ckpt)

            # 加载已保存的说话人特征
            self.speaker_features = self.load_speaker_features()

            print(f"[OpenVoice] 模型加载完成，使用设备: {device}")
            print(f"[OpenVoice] 已加载 {len(self.speaker_features)} 个说话人特征")

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
        """提取并保存说话人特征（供model_trainer调用）"""
        try:
            if not self.tone_converter:
                print(f"[OpenVoice] 转换器未初始化")
                return False

            print(f"[OpenVoice] 开始提取说话人特征: {speaker_id}")

            # 提取说话人特征
            target_se = se_extractor.get_se(
                reference_audio,
                target_dir="processed",
                tone_color_converter=self.tone_converter
            )

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
        """生成语音 - 统一接口"""
        try:
            if not self.tts_model:
                print("[OpenVoice] TTS模型未初始化")
                return None

            # 生成输出文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"static/voices/generated_{timestamp}.wav"

            if speaker_id and speaker_id in self.speaker_features:
                # 使用说话人克隆
                return self.clone_voice_with_cached_feature(text, speaker_id, output_path)
            else:
                # 使用基础TTS
                return self.synthesize_speech(text, output_path)

        except Exception as e:
            print(f"[OpenVoice] 语音生成失败: {e}")
            return None

    def clone_voice_with_cached_feature(self, text, speaker_id, output_path):
        """使用缓存的说话人特征进行语音克隆"""
        try:
            if not self.tone_converter:
                return None

            speaker_info = self.speaker_features[speaker_id]
            target_se = speaker_info['se']

            # 1. 先用基础模型生成语音
            temp_audio = f"temp_base_{speaker_id}_{int(time.time())}.wav"
            self.tts_model.tts(
                text=text,
                output_path=temp_audio,
                speaker="default",
                language="Chinese"
            )

            # 2. 使用缓存的特征进行音色转换
            self.tone_converter.convert(
                audio_src_path=temp_audio,
                src_se="base_speakers/SE/base_speakers.pth",
                tgt_se=target_se,
                output_path=output_path
            )

            # 3. 清理临时文件
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            print(f"[OpenVoice] 使用缓存特征生成语音: {speaker_id}")
            return output_path

        except Exception as e:
            print(f"[OpenVoice] 使用缓存特征生成语音失败: {e}")
            return None

    def synthesize_speech(self, text, output_path, speaker="default", language="Chinese"):
        """快速语音合成 - 模型已加载，直接使用"""
        try:
            # 直接使用已加载的模型
            self.tts_model.tts(
                text=text,
                output_path=output_path,
                speaker=speaker,
                language=language,
                speed=1.0
            )
            return output_path
        except Exception as e:
            print(f"[OpenVoice] 语音合成失败: {e}")
            return None

    def ensure_directories(self):
        """确保必要目录存在"""
        dirs = [
            "EchOfU/OpenVoice/checkpoints_v2",
            "EchOfU/OpenVoice/checkpoints/base_speakers",
            "models/OpenVoice",
            "static/voices",
            "processed"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def check_models_exist(self):
        """检查模型文件是否存在"""
        required_files = [
            "EchOfU/OpenVoice/checkpoints_v2/config.json",
            "EchOfU/OpenVoice/checkpoints_v2/checkpoints.pth",
            "EchOfU/OpenVoice/checkpoints_v2/converter.pth"
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
            zip_path = "EchOfU/OpenVoice/checkpoints_v2_0417.zip"
            extract_dir = "EchOfU/OpenVoice/"

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
    统一的语音生成接口

    Args:
        text (str): 要合成的文本
        speaker_id (str, optional): 说话人ID，如果为None则使用基础TTS

    Returns:
        str: 生成的语音文件路径，失败返回None
    """
    try:
        # 获取OpenVoice服务实例
        openvoice_service = OpenVoiceService()

        # 检查模型是否可用
        if not openvoice_service.tts_model:
            print("[OpenVoice] 模型未正确加载，请检查模型文件")
            return None

        # 生成语音
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
    提取说话人特征（供model_trainer调用）

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
