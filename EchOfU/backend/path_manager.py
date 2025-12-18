import os
from datetime import datetime


class PathManager:
    """路径管理器，统一处理所有路径相关逻辑

    提供统一的路径管理接口，支持：
    - 模型文件路径 (models/)
    - ER-NeRF训练模型路径 (models/ER-NeRF/)
    - 静态文件路径 (static/)
    - 音频文件路径 (static/voices/)
    - 视频文件路径 (static/videos/)
    - OpenVoice相关路径
    """

    def __init__(self):
        self.project_root = self._get_project_root()

    def _get_project_root(self):
        """获取项目根目录路径 - 递归向上查找EchOfU目录"""
        def find_echofu_root(start_dir):
            current_dir = os.path.abspath(start_dir)

            # 优先检查当前目录是否已经是EchOfU目录，并且包含关键的项目文件
            if os.path.basename(current_dir) == "EchOfU":
                # 检查是否包含EchOfU项目特有的目录和文件
                if (os.path.exists(os.path.join(current_dir, "backend")) and
                    os.path.exists(os.path.join(current_dir, "app.py")) and
                    (os.path.exists(os.path.join(current_dir, "models")) or
                     os.path.exists(os.path.join(current_dir, "static")))):
                    return current_dir

            # 如果当前目录包含EchOfU子目录，使用它
            echofu_subdir = os.path.join(current_dir, "EchOfU")
            if os.path.exists(echofu_subdir):
                # 检查子目录是否是真正的EchOfU项目目录
                if (os.path.exists(os.path.join(echofu_subdir, "backend")) and
                    os.path.exists(os.path.join(echofu_subdir, "app.py"))):
                    return echofu_subdir

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

    def get_root_begin_path(self, *path_parts):
        """获取根目录起始的路径"""
        return os.path.join(self.project_root, *path_parts)

    # ==================== 基础路径方法 ====================

    def get_models_path(self, *path_parts):
        """获取模型目录路径 (models/)"""
        return self.get_root_begin_path("models", *path_parts)

    def get_static_path(self, *path_parts):
        """获取静态资源目录路径 (static/)"""
        return self.get_root_begin_path("static", *path_parts)

    # ==================== ER-NeRF 相关路径 ====================

    def get_ernerf_path(self, *path_parts):
        """获取ER-NeRF模型路径 (models/ER-NeRF/)"""
        return self.get_models_path("ER-NeRF", *path_parts)

    def get_ernerf_checkpoints_path(self, *path_parts):
        """获取ER-NeRF检查点路径 (models/ER-NeRF/checkpoints/)"""
        return self.get_ernerf_path("checkpoints", *path_parts)

    def get_ernerf_config_path(self, *path_parts):
        """获取ER-NeRF配置文件路径 (models/ER-NeRF/config/)"""
        return self.get_ernerf_path("config", *path_parts)

    def get_ernerf_data_path(self, *path_parts):
        """获取ER-NeRF数据路径 (models/ER-NeRF/data/)"""
        return self.get_ernerf_path("data", *path_parts)

    def get_ernerf_model_path(self, *path_parts):
        """获取ER-NeRF训练好的模型路径 (models/ER-NeRF/)"""
        return self.get_ernerf_path(*path_parts)

    # ==================== OpenVoice 相关路径 ====================

    def get_openvoice_v2_path(self, *path_parts):
        """获取OpenVoice V2相关路径"""
        return self.get_root_begin_path("OpenVoice/checkpoints_v2", *path_parts)

    def get_openvoice_model_path(self, *path_parts):
        """获取OpenVoice模型路径 (models/OpenVoice/)"""
        return self.get_models_path("OpenVoice", *path_parts)

    def get_speaker_features_path(self):
        """获取说话人特征文件路径 (models/OpenVoice/speaker_features.json)"""
        return self.get_openvoice_model_path("speaker_features.json")

    def get_speaker_feature_tensor_path(self, speaker_id):
        """获取指定说话人的特征张量文件路径"""
        return self.get_openvoice_model_path(f"{speaker_id}_se.pth")

    def get_processed_path(self, *path_parts):
        """获取处理后的数据路径"""
        return self.get_root_begin_path("processed", *path_parts)

    # ==================== 音频文件路径 ====================

    def get_voices_path(self, *path_parts):
        """获取音频目录路径 (static/voices/)"""
        return self.get_static_path("voices", *path_parts)

    def get_ref_voice_path(self, filename=None):
        """获取参考音频文件路径 (static/voices/ref_voices/)"""
        ref_dir = self.get_voices_path("ref_voices")
        if filename:
            return os.path.join(ref_dir, filename)
        return ref_dir

    def get_res_voice_path(self, filename=None):
        """获取生成音频文件路径 (static/voices/res_voices/)"""
        res_dir = self.get_voices_path("res_voices")
        if filename:
            return os.path.join(res_dir, filename)
        return res_dir

    def get_output_voice_path(self, timestamp):
        """生成输出语音文件路径 (带时间戳)"""
        return self.get_res_voice_path(f"generated_{timestamp}.wav")

    def get_extracted_voice_path(self, timestamp):
        """生成提取音频文件路径 (带时间戳)"""
        return self.get_ref_voice_path(f"extracted_{timestamp}.wav")

    def get_temp_voice_path(self, base_name):
        """生成临时音频文件路径"""
        return os.path.join(self.project_root, f"temp_{base_name}_{int(datetime.now().timestamp())}.wav")

    # ==================== 视频文件路径 ====================

    def get_videos_path(self, *path_parts):
        """获取视频目录路径 (static/videos/)"""
        return self.get_static_path("videos", *path_parts)

    def get_ref_video_path(self, filename=None):
        """获取参考视频文件路径 (static/videos/ref_videos/)"""
        ref_dir = self.get_videos_path("ref_videos")
        if filename:
            return os.path.join(ref_dir, filename)
        return ref_dir

    def get_res_video_path(self, filename=None):
        """获取生成视频文件路径 (static/videos/res_videos/)"""
        res_dir = self.get_videos_path("res_videos")
        if filename:
            return os.path.join(res_dir, filename)
        return res_dir

    def get_output_video_path(self, timestamp, extension="mp4"):
        """生成输出视频文件路径 (带时间戳)"""
        return self.get_res_video_path(f"generated_{timestamp}.{extension}")

    def get_extracted_video_path(self, timestamp, extension="mp4"):
        """生成提取视频文件路径 (带时间戳)"""
        return self.get_ref_video_path(f"extracted_{timestamp}.{extension}")

    # ==================== 通用工具方法 ====================

    def ensure_directory(self, directory_path):
        """确保目录存在，不存在则创建"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def ensure_file_directory(self, file_path):
        """确保文件的目录存在"""
        directory = os.path.dirname(file_path)
        return self.ensure_directory(directory)

    def get_unique_filename(self, base_path, extension=""):
        """获取唯一的文件名，避免冲突"""
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        directory = os.path.dirname(base_path)

        if extension and not extension.startswith('.'):
            extension = '.' + extension

        counter = 1
        final_path = base_path + extension

        while os.path.exists(final_path):
            name_with_counter = f"{base_name}_{counter}"
            final_path = os.path.join(directory, name_with_counter + extension)
            counter += 1

        return final_path

    def path_exists(self, path):
        """检查路径是否存在"""
        return os.path.exists(path)

    def is_file(self, path):
        """检查路径是否为文件"""
        return os.path.isfile(path)

    def is_directory(self, path):
        """检查路径是否为目录"""
        return os.path.isdir(path)

    def get_project_info(self):
        """获取项目路径信息摘要"""
        return {
            "project_root": self.project_root,
            "models_dir": self.get_models_path(),
            "ernerf_dir": self.get_ernerf_path(),
            "openvoice_dir": self.get_openvoice_model_path(),
            "voices_dir": self.get_voices_path(),
            "videos_dir": self.get_videos_path(),
        }