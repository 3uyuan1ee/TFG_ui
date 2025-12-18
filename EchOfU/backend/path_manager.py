import os


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

            # 如果当前目录包含EchOfU子目录，使用它
            echofu_subdir = os.path.join(current_dir, "EchOfU")
            if os.path.exists(echofu_subdir) and os.path.exists(os.path.join(echofu_subdir, "OpenVoice")):
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

    def get_openvoice_v2_path(self, *path_parts):
        """获取OpenVoice V2相关路径"""
        return self.get_root_begin_path("OpenVoice/checkpoints_v2", *path_parts)

    def get_speaker_features_path(self):
        """获取说话人特征文件路径"""
        return os.path.join(self.project_root, "models/OpenVoice/speaker_features.json")

    def get_ref_voice_path(self, filename=None):
        """获取参考音频文件路径"""
        if filename:
            return os.path.join(self.project_root, "static/voices/ref_voices", filename)
        return os.path.join(self.project_root, "static/voices/ref_voices")

    def get_res_voice_path(self, filename=None):
        """获取生成音频文件路径"""
        if filename:
            return os.path.join(self.project_root, "static/voices/res_voices", filename)
        return os.path.join(self.project_root, "static/voices/res_voices")

    def get_output_voice_path(self, timestamp):
        """生成输出语音文件路径"""
        return self.get_res_voice_path(f"generated_{timestamp}.wav")