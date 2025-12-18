"""
文件管理器 - 处理文件上传、列表和管理功能
与PathManager集成，确保正确的路径处理
"""

import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import request, jsonify
from .path_manager import PathManager


class FileManager:
    """文件管理器类，处理音频和视频文件的上传、列表和管理"""

    def __init__(self):
        self.path_manager = PathManager()
        # 确保必要目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录结构存在"""
        directories = [
            self.path_manager.get_ref_voice_path(),
            self.path_manager.get_res_voice_path(),
            self.path_manager.get_ref_video_path(),
            self.path_manager.get_res_video_path()
        ]

        for directory in directories:
            self.path_manager.ensure_directory(directory)

    def get_supported_audio_extensions(self):
        """获取支持的音频文件扩展名"""
        return {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

    def get_supported_video_extensions(self):
        """获取支持的视频文件扩展名"""
        return {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}

    def _generate_safe_filename(self, original_filename):
        """生成安全的文件名"""
        # 获取文件名和扩展名
        base_name = os.path.splitext(original_filename)[0]
        extension = os.path.splitext(original_filename)[1].lower()

        # 清理文件名（移除特殊字符）
        safe_base_name = ''.join(c for c in base_name if c.isalnum() or c in '._-')

        # 生成唯一文件名（添加时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{safe_base_name}_{timestamp}{extension}"

        return safe_filename

    def _validate_file(self, file, allowed_extensions, max_size_mb=100):
        """验证上传的文件"""
        # 检查文件类型
        filename = file.filename
        if not filename:
            return False, "没有选择文件"

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return False, f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"

        # 检查文件大小
        max_size = max_size_mb * 1024 * 1024
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0, os.SEEK_SET)

        if file_size > max_size:
            return False, f"文件大小超过限制。最大支持{max_size_mb}MB，当前文件大小: {file_size / (1024*1024):.2f}MB"

        return True, "文件验证通过"

    def _get_file_info(self, file_path):
        """获取文件信息"""
        if not os.path.exists(file_path):
            return None

        stat = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'file_type': os.path.splitext(file_path)[1][1:].upper()  # 去掉点号，转为大写
        }

    def _get_relative_path(self, absolute_path):
        """将绝对路径转换为相对路径"""
        if absolute_path.startswith(self.path_manager.project_root):
            relative_path = absolute_path[len(self.path_manager.project_root):].lstrip('/\\')
            return relative_path
        return absolute_path

    # ==================== 参考音频文件管理 ====================

    def upload_reference_audio(self, audio_file):
        """上传参考音频文件"""
        # 验证文件
        is_valid, message = self._validate_file(
            audio_file,
            self.get_supported_audio_extensions()
        )

        if not is_valid:
            return {
                'status': 'error',
                'message': message
            }, 400

        # 生成安全文件名
        safe_filename = self._generate_safe_filename(audio_file.filename)

        # 保存文件
        save_path = self.path_manager.get_ref_voice_path(safe_filename)
        audio_file.save(save_path)

        # 获取相对路径
        relative_path = self._get_relative_path(save_path)

        print(f"[FileManager] 参考音频上传成功: {audio_file.filename} -> {relative_path}")

        return {
            'status': 'success',
            'message': '音频上传成功',
            'filename': safe_filename,
            'relative_path': relative_path,
            'original_name': audio_file.filename
        }

    def get_reference_audios(self):
        """获取参考音频文件列表"""
        ref_voices_dir = self.path_manager.get_ref_voice_path()

        if not os.path.exists(ref_voices_dir):
            return {
                'status': 'success',
                'files': [],
                'total_count': 0,
                'message': '参考音频目录不存在'
            }

        audio_files = []
        audio_extensions = self.get_supported_audio_extensions()

        for filename in os.listdir(ref_voices_dir):
            file_path = os.path.join(ref_voices_dir, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in audio_extensions:
                    file_info = self._get_file_info(file_path)
                    if file_info:
                        file_info['relative_path'] = self._get_relative_path(file_path)
                        audio_files.append(file_info)

        # 按修改时间降序排列（最新的在前）
        audio_files.sort(key=lambda x: x['modified_time'], reverse=True)

        print(f"[FileManager] 获取到 {len(audio_files)} 个参考音频文件")

        return {
            'status': 'success',
            'files': audio_files,
            'total_count': len(audio_files)
        }

    # ==================== 训练视频文件管理 ====================

    def upload_training_video(self, video_file):
        """上传训练视频文件"""
        # 验证文件
        is_valid, message = self._validate_file(
            video_file,
            self.get_supported_video_extensions(),
            max_size_mb=200  # 视频文件允许更大一些
        )

        if not is_valid:
            return {
                'status': 'error',
                'message': message
            }, 400

        # 生成安全文件名
        safe_filename = self._generate_safe_filename(video_file.filename)

        # 保存文件
        save_path = self.path_manager.get_ref_video_path(safe_filename)
        video_file.save(save_path)

        # 获取相对路径
        relative_path = self._get_relative_path(save_path)

        print(f"[FileManager] 训练视频上传成功: {video_file.filename} -> {relative_path}")

        return {
            'status': 'success',
            'message': '视频上传成功',
            'filename': safe_filename,
            'relative_path': relative_path,
            'original_name': video_file.filename
        }

    def get_training_videos(self):
        """获取训练视频文件列表"""
        ref_videos_dir = self.path_manager.get_ref_video_path()

        if not os.path.exists(ref_videos_dir):
            return {
                'status': 'success',
                'files': [],
                'total_count': 0,
                'message': '参考视频目录不存在'
            }

        video_files = []
        video_extensions = self.get_supported_video_extensions()

        for filename in os.listdir(ref_videos_dir):
            file_path = os.path.join(ref_videos_dir, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in video_extensions:
                    file_info = self._get_file_info(file_path)
                    if file_info:
                        file_info['relative_path'] = self._get_relative_path(file_path)
                        video_files.append(file_info)

        # 按修改时间降序排列（最新的在前）
        video_files.sort(key=lambda x: x['modified_time'], reverse=True)

        print(f"[FileManager] 获取到 {len(video_files)} 个训练视频文件")

        return {
            'status': 'success',
            'files': video_files,
            'total_count': len(video_files)
        }

    # ==================== 模型文件管理 ====================

    def get_available_models(self):
        """获取可用模型列表"""
        available_models = []

        # # 获取SyncTalk模型
        # synctalk_dir = self.path_manager.get_models_path("SyncTalk")
        # if os.path.exists(synctalk_dir):
        #     for item in os.listdir(synctalk_dir):
        #         item_path = os.path.join(synctalk_dir, item)
        #         if os.path.isdir(item_path):
        #             model_files = [f for f in os.listdir(item_path)
        #                         if f.endswith(('.pth', '.ckpt', '.pt', '.bin', '.safetensors'))]
        #             if model_files:
        #                 stat = os.stat(item_path)
        #                 available_models.append({
        #                     'name': item,
        #                     'type': 'SyncTalk',
        #                     'path': f"models/SyncTalk/{item}",
        #                     'model_files_count': len(model_files),
        #                     'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
        #                     'description': f'SyncTalk模型 - 包含{len(model_files)}个模型文件'
        #                 })

        # 获取ER-NeRF模型
        ernef_dir = self.path_manager.get_ernerf_path()
        if os.path.exists(ernef_dir):
            # 首先检查子目录中的模型
            for item in os.listdir(ernef_dir):
                item_path = os.path.join(ernef_dir, item)
                if os.path.isdir(item_path):
                    model_files = [f for f in os.listdir(item_path)
                                if f.endswith(('.pth', '.ckpt', '.pt', '.bin', '.safetensors'))]
                    if model_files:
                        stat = os.stat(item_path)
                        available_models.append({
                            'name': item,
                            'type': 'ER-NeRF',
                            'path': f"models/ER-NeRF/{item}",
                            'model_files_count': len(model_files),
                            'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                            'description': f'ER-NeRF模型 - 包含{len(model_files)}个模型文件'
                        })

            # 然后检查直接在ER-NeRF目录下的模型文件
            model_files = [f for f in os.listdir(ernef_dir)
                          if f.endswith(('.pth', '.ckpt', '.pt', '.bin', '.safetensors')) and not f.startswith('.')]
            if model_files:
                stat = os.stat(ernef_dir)
                available_models.append({
                    'name': 'root',
                    'type': 'ER-NeRF',
                    'path': 'models/ER-NeRF',
                    'model_files_count': len(model_files),
                    'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'description': f'ER-NeRF根目录模型 - 包含{len(model_files)}个模型文件'
                })

        # 添加默认选项（如果没有找到任何模型）
        if not available_models:
            available_models.extend([
                {
                    'name': 'default',
                    'type': 'SyncTalk',
                    'path': 'models/SyncTalk/default',
                    'model_files_count': 0,
                    'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'description': '默认SyncTalk模型（需要手动配置）'
                },
                {
                    'name': 'default',
                    'type': 'ER-NeRF',
                    'path': 'models/ER-NeRF/default',
                    'model_files_count': 0,
                    'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'description': '默认ER-NeRF模型（需要手动配置）'
                }
            ])

        # 按类型和名称排序
        available_models.sort(key=lambda x: (x['type'], x['name']))

        print(f"[FileManager] 获取到 {len(available_models)} 个可用模型")

        return {
            'status': 'success',
            'models': available_models,
            'total_count': len(available_models)
        }

    def get_model_details(self, model_type, model_name):
        """获取模型详细信息"""
        # 根据模型类型确定基础路径
        if model_type == 'SyncTalk':
            base_path = self.path_manager.get_models_path("SyncTalk", model_name)
        elif model_type == 'ER-NeRF':
            base_path = self.path_manager.get_ernerf_path(model_name)
        else:
            return {
                'status': 'error',
                'message': f'不支持的模型类型: {model_type}'
            }, 400

        if not os.path.exists(base_path):
            return {
                'status': 'error',
                'message': f'模型目录不存在: {base_path}'
            }, 404

        model_files = []
        total_size = 0

        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_path)
                stat = os.stat(file_path)

                file_size = stat.st_size
                total_size += file_size

                # 判断文件类型
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in ['.pth', '.ckpt', '.pt', '.bin', '.safetensors']:
                    file_type = 'model_weight'
                elif file_ext in ['.json', '.yaml', '.yml', '.txt']:
                    file_type = 'config'
                elif file_ext in ['.py']:
                    file_type = 'code'
                else:
                    file_type = 'other'

                model_files.append({
                    'filename': file,
                    'relative_path': relative_path,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'file_type': file_type,
                    'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

        # 获取目录信息
        dir_stat = os.stat(base_path)

        model_details = {
            'name': model_name,
            'type': model_type,
            'path': base_path,
            'total_files': len(model_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'created_time': datetime.fromtimestamp(dir_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_time': datetime.fromtimestamp(dir_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'files': model_files
        }

        print(f"[FileManager] 获取模型详细信息: {model_type}/{model_name} - {len(model_files)}个文件")

        return {
            'status': 'success',
            'model_details': model_details
        }


# 全局实例
file_manager = FileManager()