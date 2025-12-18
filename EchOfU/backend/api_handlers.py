"""
API路由处理器 - 处理文件相关的API路由
与FileManager集成，提供统一的API接口
"""

from flask import request, jsonify
from .file_manager import file_manager


def upload_reference_audio():
    """上传参考音频文件API"""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有音频文件'
            }), 400

        audio_file = request.files['audio']
        result = file_manager.upload_reference_audio(audio_file)

        # 检查是否返回了(status_code, data)元组
        if isinstance(result, tuple) and len(result) == 2:
            data, status_code = result
            return jsonify(data), status_code

        # 否则直接返回结果
        return jsonify(result)

    except Exception as e:
        print(f"[API] 参考音频上传失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'音频上传失败: {str(e)}'
        }), 500


def get_reference_audios():
    """获取参考音频文件列表API"""
    try:
        result = file_manager.get_reference_audios()
        return jsonify(result)

    except Exception as e:
        print(f"[API] 获取参考音频列表失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'files': [],
            'total_count': 0
        }), 500


def upload_training_video():
    """上传训练视频文件API"""
    try:
        if 'video' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有视频文件'
            }), 400

        video_file = request.files['video']
        result = file_manager.upload_training_video(video_file)

        # 检查是否返回了(status_code, data)元组
        if isinstance(result, tuple) and len(result) == 2:
            data, status_code = result
            return jsonify(data), status_code

        # 否则直接返回结果
        return jsonify(result)

    except Exception as e:
        print(f"[API] 训练视频上传失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'视频上传失败: {str(e)}'
        }), 500


def get_training_videos():
    """获取训练视频文件列表API"""
    try:
        result = file_manager.get_training_videos()
        return jsonify(result)

    except Exception as e:
        print(f"[API] 获取训练视频列表失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'files': [],
            'total_count': 0
        }), 500


def get_available_models():
    """获取可用模型列表API"""
    try:
        result = file_manager.get_available_models()
        return jsonify(result)

    except Exception as e:
        print(f"[API] 获取可用模型列表失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'models': [],
            'total_count': 0
        }), 500


def get_model_details(model_type, model_name):
    """获取模型详细信息API"""
    try:
        result = file_manager.get_model_details(model_type, model_name)

        # 检查是否返回了(status_code, data)元组
        if isinstance(result, tuple) and len(result) == 2:
            data, status_code = result
            return jsonify(data), status_code

        # 否则直接返回结果
        return jsonify(result)

    except Exception as e:
        print(f"[API] 获取模型详细信息失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500