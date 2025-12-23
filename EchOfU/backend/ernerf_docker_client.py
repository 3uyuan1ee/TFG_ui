"""
ER-NeRF Docker 客户端
用于从Backend调用Docker化的ER-NeRF服务

使用方式:
    from backend.ernerf_docker_client import ERNeRFDockerClient

    client = ERNeRFDockerClient()

    # 预处理
    success = client.preprocess(video_path, task_id)

    # 训练
    success = client.train(data_path, model_path, gpu_id=0)

    # 推理
    video_path = client.infer(data_path, model_path, audio_npy_path)
"""

import os
import subprocess
import shutil
import time
from typing import Optional, Tuple
from .path_manager import PathManager


class ERNeRFDockerClient:
    """
    ER-NeRF Docker 客户端

    封装所有与ER-NeRF Docker容器交互的逻辑
    """

    def __init__(self,
                 docker_compose_file: str = "docker-compose.ernerf.yml",
                 service_name: str = "ernerf",
                 project_root: Optional[str] = None):
        """
        初始化客户端

        Args:
            docker_compose_file: docker-compose文件名
            service_name: docker服务名称
            project_root: 项目根目录（默认自动检测）
        """
        self.pm = PathManager()
        self.docker_compose_file = docker_compose_file
        self.service_name = service_name

        # 获取项目根目录（EchOfU/）
        if project_root:
            self.project_root = project_root
        else:
            # 自动检测：假设当前文件在 EchOfU/backend/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(current_dir)

        # docker-compose完整路径
        self.docker_compose_path = os.path.join(self.project_root, docker_compose_file)

        print(f"[ERNeRFDockerClient] 初始化完成")
        print(f"[ERNeRFDockerClient] 项目根目录: {self.project_root}")
        print(f"[ERNeRFDockerClient] Docker Compose: {self.docker_compose_path}")

    def _run_docker_command(self,
                           mode: str,
                           args: list,
                           capture_output: bool = True) -> Tuple[bool, str]:
        """
        运行Docker命令

        Args:
            mode: entrypoint模式 (preprocess, train, test, etc.)
            args: 传递给entrypoint的参数列表
            capture_output: 是否捕获输出

        Returns:
            (success, output)
        """
        cmd = [
            "docker", "compose",
            "-f", self.docker_compose_path,
            "run", "--rm",
            self.service_name,
            mode
        ] + args

        print(f"[ERNeRFDockerClient] 执行命令: {' '.join(cmd)}")

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self.project_root
                )
                return True, result.stdout
            else:
                subprocess.run(cmd, check=True, cwd=self.project_root)
                return True, ""

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            print(f"[ERNeRFDockerClient] 命令执行失败: {error_msg}")
            return False, error_msg
        except Exception as e:
            print(f"[ERNeRFDockerClient] 未知错误: {e}")
            return False, str(e)

    def check_container_running(self) -> bool:
        """
        检查ER-NeRF容器是否在运行

        Returns:
            bool: 容器是否运行中
        """
        try:
            cmd = [
                "docker", "compose",
                "-f", self.docker_compose_path,
                "ps", "-q", self.service_name
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return bool(result.stdout.strip())
        except:
            return False

    def start_container(self) -> bool:
        """
        启动ER-NeRF容器（后台运行）

        Returns:
            bool: 是否成功启动
        """
        print(f"[ERNeRFDockerClient] 启动容器...")

        try:
            cmd = [
                "docker", "compose",
                "-f", self.docker_compose_path,
                "up", "-d"
            ]
            subprocess.run(cmd, check=True, cwd=self.project_root)
            print(f"[ERNeRFDockerClient] 容器启动成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERNeRFDockerClient] 容器启动失败: {e}")
            return False

    def preprocess(self,
                  video_path: str,
                  task_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        数据预处理

        Args:
            video_path: 输入视频路径
            task_id: 任务ID（可选，默认使用视频文件名）

        Returns:
            (success, message)
        """
        print(f"[ERNeRFDockerClient] 开始数据预处理")
        print(f"[ERNeRFDockerClient] 视频: {video_path}")
        print(f"[ERNeRFDockerClient] 任务ID: {task_id or '自动'}")

        # 检查视频文件
        if not os.path.exists(video_path):
            return False, f"视频文件不存在: {video_path}"

        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(video_path):
            video_path = os.path.abspath(video_path)

        args = [video_path]
        if task_id:
            args.append(task_id)

        success, output = self._run_docker_command("preprocess", args, capture_output=True)

        if success:
            print(f"[ERNeRFDockerClient] 预处理完成")
            # 返回数据路径
            data_path = os.path.join(self.project_root, "data", task_id or os.path.basename(video_path).split('.')[0])
            return True, f"预处理完成，数据保存至: {data_path}"
        else:
            return False, f"预处理失败: {output}"

    def train(self,
              data_path: str,
              model_path: str,
              gpu_id: int = 0,
              stage: str = "auto") -> Tuple[bool, str]:
        """
        模型训练

        Args:
            data_path: 数据集路径（相对于workspace/data）
            model_path: 模型保存路径（相对于workspace/models/ER-NeRF）
            gpu_id: GPU编号
            stage: 训练阶段 (auto|head|lips|torso)

        Returns:
            (success, message)
        """
        print(f"[ERNeRFDockerClient] 开始模型训练")
        print(f"[ERNeRFDockerClient] 数据: {data_path}")
        print(f"[ERNeRFDockerClient] 模型: {model_path}")
        print(f"[ERNeRFDockerClient] GPU: {gpu_id}")
        print(f"[ERNeRFDockerClient] 阶段: {stage}")

        args = [
            data_path,
            model_path,
            str(gpu_id),
            stage
        ]

        # 训练可能需要很长时间，不捕获输出以实时显示日志
        # 或者使用日志文件重定向
        success, output = self._run_docker_command("train", args, capture_output=False)

        if success:
            print(f"[ERNeRFDockerClient] 训练完成")
            return True, f"训练完成，模型保存至: {model_path}"
        else:
            return False, f"训练失败"

    def infer(self,
              data_path: str,
              model_path: str,
              audio_npy_path: str,
              with_torso: bool = True) -> Tuple[bool, str]:
        """
        模型推理

        Args:
            data_path: 数据集路径
            model_path: 模型路径
            audio_npy_path: 音频特征文件路径（.npy）
            with_torso: 是否包含躯干

        Returns:
            (success, video_path_or_error_message)
        """
        print(f"[ERNeRFDockerClient] 开始模型推理")
        print(f"[ERNeRFDockerClient] 数据: {data_path}")
        print(f"[ERNeRFDockerClient] 模型: {model_path}")
        print(f"[ERNeRFDockerClient] 音频特征: {audio_npy_path}")
        print(f"[ERNeRFDockerClient] 包含躯干: {with_torso}")

        # 检查音频特征文件
        if not os.path.exists(audio_npy_path):
            return False, f"音频特征文件不存在: {audio_npy_path}"

        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(audio_npy_path):
            audio_npy_path = os.path.abspath(audio_npy_path)

        args = [
            data_path,
            model_path,
            audio_npy_path,
            "true" if with_torso else "false"
        ]

        success, output = self._run_docker_command("test", args, capture_output=True)

        if success:
            # 查找生成的视频文件
            video_path = self._find_result_video(model_path)
            if video_path:
                print(f"[ERNeRFDockerClient] 推理完成: {video_path}")
                return True, video_path
            else:
                return False, "推理完成但未找到生成的视频文件"
        else:
            return False, f"推理失败: {output}"

    def extract_audio_features(self,
                               wav_path: str) -> Tuple[bool, str]:
        """
        提取音频特征（DeepSpeech）

        Args:
            wav_path: WAV音频文件路径

        Returns:
            (success, npy_path_or_error_message)
        """
        print(f"[ERNeRFDockerClient] 提取音频特征")
        print(f"[ERNeRFDockerClient] 音频: {wav_path}")

        # 检查音频文件
        if not os.path.exists(wav_path):
            return False, f"音频文件不存在: {wav_path}"

        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(wav_path):
            wav_path = os.path.abspath(wav_path)

        args = [wav_path]

        success, output = self._run_docker_command("extract-features", args, capture_output=True)

        if success:
            # 返回.npy文件路径
            npy_path = wav_path.replace('.wav', '.npy')
            if os.path.exists(npy_path):
                return True, npy_path
            else:
                return False, "特征提取完成但未找到输出文件"
        else:
            return False, f"特征提取失败: {output}"

    def _find_result_video(self, model_path: str) -> Optional[str]:
        """
        查找生成的视频文件

        Args:
            model_path: 模型路径

        Returns:
            视频文件路径或None
        """
        # 可能的结果目录
        possible_result_dirs = [
            os.path.join(model_path, "results"),
            os.path.join(self.project_root, "results"),
            os.path.join(self.project_root, "ER-NeRF", "results"),
        ]

        for results_dir in possible_result_dirs:
            if os.path.exists(results_dir):
                # 查找最新的mp4文件
                mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                if mp4_files:
                    latest_video = max(
                        mp4_files,
                        key=lambda f: os.path.getctime(os.path.join(results_dir, f))
                    )
                    return os.path.join(results_dir, latest_video)

        return None

    def get_data_path(self, task_id: str) -> str:
        """
        获取数据路径

        Args:
            task_id: 任务ID

        Returns:
            数据路径
        """
        return os.path.join(self.project_root, "data", task_id)

    def get_model_path(self, task_id: str) -> str:
        """
        获取模型路径

        Args:
            task_id: 任务ID

        Returns:
            模型路径
        """
        return os.path.join(self.project_root, "models", "ER-NeRF", task_id)


# 便捷函数：创建全局单例
_global_client = None

def get_ernerf_docker_client() -> ERNeRFDockerClient:
    """
    获取ER-NeRF Docker客户端全局单例

    Returns:
        ERNeRFDockerClient实例
    """
    global _global_client
    if _global_client is None:
        _global_client = ERNeRFDockerClient()
    return _global_client
