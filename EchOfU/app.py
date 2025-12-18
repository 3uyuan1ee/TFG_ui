# =============================================================================
# SYNCTALK AI - 智能语音对话系统
# =============================================================================
# 应用主文件：负责Flask应用的路由配置和HTTP请求处理
#
# 功能模块：
# - 首页展示 (index)
# - 视频生成 (video_generation)
# - 模型训练 (model_training)
# - 音频克隆 (audio_clone)
# - 人机对话 (chat_system)
# - 系统状态监控 (API接口)
# =============================================================================

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from datetime import datetime

from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response
from backend.voice_generator import OpenVoiceService
from backend.api_handlers import (
    upload_reference_audio as api_upload_reference_audio,
    get_reference_audios as api_get_reference_audios,
    upload_training_video as api_upload_training_video,
    get_training_videos as api_get_training_videos,
    get_available_models as api_get_available_models,
    get_model_details as api_get_model_details
)
import psutil

# =============================================================================
# GPU监控模块初始化
# =============================================================================
# 尝试导入 GPUtil 库用于GPU监控，如果未安装则提供空实现
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    class GPUtil:
        @staticmethod
        def getGPUs():
            return []

# =============================================================================
# Flask应用初始化
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为100MB

# 确保必要的目录结构存在（基本目录）
for folder in [ 'EchOfU/static/audios', 'EchOfU/static/videos',  'EchOfU/static/history']:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# 页面路由
# =============================================================================

@app.route('/')
def index():
    """首页路由 - 展示系统导航卡片"""
    return render_template('index.html')

@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    """视频生成页面路由 - 处理音频驱动的视频合成请求"""

    if request.method == 'POST':
        # ==================== POST请求：处理视频生成 ====================
        try:
            # 收集表单数据

            # ToDo : 这里也是一样，后续需要调整参数，参考音频路径以及改为参考音频编号
            data = {
                "model_name": request.form.get('model_name', 'SyncTalk'),    # 生成模型选择
                "model_param": request.form.get('model_param', ''),          # 模型参数路径
                "ref_audio": request.form.get('ref_audio', ''),              # 参考音频路径
                "gpu_choice": request.form.get('gpu_choice', 'GPU0'),        # GPU设备选择
                "target_text": request.form.get('target_text', '')           # 目标文本内容
            }

            # 调用后端视频生成模块
            video_path = generate_video(data)

            # ==================== 历史记录保存 ====================
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': data['model_name'],
                'params': data,
                'output': video_path
            }

            # 保存到历史记录文件
            history_file = 'EchOfU/static/history/video_generation.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(history_entry)

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

            return jsonify({
                'status': 'success',
                'video_path': video_path,
                'message': '视频生成成功'
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    # ==================== GET请求：渲染页面模板 ====================
    try:
        # 获取可用GPU设备列表
        if GPU_AVAILABLE:
            gpu_list = [f"GPU{i}" for i in range(len(GPUtil.getGPUs()))]
        else:
            gpu_list = ['GPU0']  # 默认提供GPU0选项
    except:
        gpu_list = ['GPU0']

    # 获取可用模型列表
    models = ['SyncTalk', 'ER-NeRF']
    synctalk_dir = './SyncTalk/model'
    if os.path.exists(synctalk_dir):
        for item in os.listdir(synctalk_dir):
            if os.path.isdir(os.path.join(synctalk_dir, item)):
                models.append(item)

    return render_template('video_generation.html', gpus=gpu_list, models=models)

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    """模型训练页面路由 - 处理深度学习模型训练请求"""

    if request.method == 'POST':
        # ==================== POST请求：处理模型训练 ====================
        try:
            # 收集训练参数
            data = {
                "model_choice": request.form.get('model_choice', 'SyncTalk'),     # 模型类型选择
                "ref_video": request.form.get('ref_video', ''),                  # 参考视频路径
                "gpu_choice": request.form.get('gpu_choice', 'GPU0'),            # 训练GPU选择
                "epoch": request.form.get('epoch', '100'),                       # 训练轮数
                "custom_params": request.form.get('custom_params', '')           # 自定义参数
            }

            # 调用后端模型训练模块
            result = train_model(data)

            return jsonify({
                'status': 'success',
                'message': '模型训练开始',
                'task_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            })

            # ToDo : 这里不够完善，后续可以返回训练日志或进度
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    # GET请求：渲染模型训练页面
    return render_template('model_training.html')

@app.route('/audio_clone', methods=['GET', 'POST'])
def audio_clone():
    """音频克隆页面路由 - 处理语音克隆请求"""

    if request.method == 'POST':
        # ==================== POST请求：处理音频克隆 ====================
        try:
            # 收集音频克隆参数
            data = {
                "original_audio_path": request.form.get('original_audio_path', ''),  # 原始音频文件路径
                "audio_id": request.form.get('audio_id', ''),                        # 源音频ID
                "target_audio_id": request.form.get('target_audio_id', ''),          # 目标音频ID
                "gen_audio_id": request.form.get('gen_audio_id', ''),                # 生成音频ID
                "generate_text": request.form.get('generate_text', '')               # 生成文本内容
            }

            # 验证必要参数 - 区分克隆模式和生成模式
            if data['generate_text']:
                # 生成模式：只需要音频ID和文本
                if not data['audio_id']:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必要参数：音频ID'
                    }), 400
                if not data['generate_text'].strip():
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必要参数：生成文本'
                    }), 400
            else:
                # 克隆模式：需要原音频路径和目标音频ID
                if not data['original_audio_path'] or not data['target_audio_id']:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必要参数：原音频路径和目标音频ID'
                    }), 400

            # 创建OpenVoice服务实例（单例模式）
            ov_service = OpenVoiceService()

            if data['generate_text']:
                # ==================== 生成模式：使用已有特征生成音频 ====================
                audio_id = data['audio_id']  # 使用音频ID而不是target_audio_id
                generate_text = data['generate_text']

                print(f"[音频生成] 开始生成音频:")
                print(f"[音频生成] 音频ID: {audio_id}")
                print(f"[音频生成] 生成文本: {generate_text}")

                # 直接使用已有特征生成音频
                generated_audio_path = ov_service.generate_speech(
                    text=generate_text,
                    speaker_id=audio_id
                )

                if generated_audio_path:
                    # 转换为相对路径供前端使用
                    # 获取当前Flask应用文件所在目录作为基准
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if generated_audio_path.startswith(current_dir):
                        relative_path = generated_audio_path[len(current_dir):].lstrip('/\\')
                    else:
                        # 如果不是以当前目录开头，直接使用文件名部分
                        relative_path = os.path.basename(generated_audio_path)
                        # 如果文件在static/voices/res_voices下，保留路径
                        if 'static/voices/res_voices' in generated_audio_path:
                            parts = generated_audio_path.split('static/voices/res_voices')
                            if len(parts) > 1:
                                relative_path = f"static/voices/res_voices{parts[1]}"

                    print(f"[音频生成] 路径转换: {generated_audio_path} -> {relative_path}")

                    return jsonify({
                        'status': 'success',
                        'message': '音频生成成功',
                        'cloned_audio_path': relative_path
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '语音生成失败'
                    }), 500

            else:
                # ==================== 克隆模式：提取说话人特征 ====================
                # 将相对路径转换为绝对路径
                original_audio_path = data['original_audio_path']
                if not os.path.isabs(original_audio_path):
                    # 获取当前Flask应用的目录作为基准
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    original_audio_path = os.path.join(current_dir, original_audio_path)
                    original_audio_path = os.path.normpath(original_audio_path)

                print(f"[音频克隆] 开始处理克隆请求:")
                print(f"[音频克隆] 原音频路径: {data['original_audio_path']} -> {original_audio_path}")
                print(f"[音频克隆] 目标音频ID: {data['target_audio_id']}")

                # 提取说话人特征
                if not ov_service.extract_and_save_speaker_feature(
                    speaker_id=data['target_audio_id'],
                    reference_audio=original_audio_path
                ):
                    return jsonify({
                        'status': 'error',
                        'message': '说话人特征提取失败'
                    }), 500

                # 克隆模式：只进行特征提取，不生成音频
                return jsonify({
                    'status': 'success',
                    'message': '说话人特征提取完成，可以用于后续音频生成'
                })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    # GET请求：渲染音频克隆页面
    return render_template('audio_clone.html')

# =============================================================================
# API接口路由
# =============================================================================

@app.route('/api/cloned-audios', methods=['GET'])
def get_cloned_audios():
    """获取已克隆的音频列表API - 为页面提供数据"""
    try:
        # 使用OpenVoiceService获取实际已保存的说话人特征
        ov_service = OpenVoiceService()
        available_speakers = ov_service.list_available_speakers()

        # 获取说话人特征信息
        speaker_features = ov_service.speaker_features
        cloned_audios = []

        for speaker_id in available_speakers:
            if speaker_id in speaker_features:
                feature_info = speaker_features[speaker_id]
                cloned_audios.append({
                    "id": speaker_id,
                    "name": speaker_id,
                    "created_at": feature_info.get('created_time', '未知时间'),
                    "reference_audio": feature_info.get('reference_audio', '未知'),
                    "status": "已提取特征"
                })

        print(f"[API] 获取到 {len(cloned_audios)} 个已克隆的音频")
        return jsonify({
            'status': 'success',
            'audios': cloned_audios,
            'total_count': len(cloned_audios)
        })

    except Exception as e:
        print(f"[API] 获取已克隆音频列表失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'audios': []
        }), 500

@app.route('/api/upload-reference-audio', methods=['POST'])
def upload_reference_audio():
    """上传参考音频文件API - 使用backend模块"""
    return api_upload_reference_audio()

@app.route('/api/reference-audios', methods=['GET'])
def get_reference_audios():
    """获取参考音频文件列表API - 使用backend模块"""
    return api_get_reference_audios()

@app.route('/api/upload-training-video', methods=['POST'])
def upload_training_video():
    """上传训练视频文件API - 使用backend模块"""
    return api_upload_training_video()

@app.route('/api/training-videos', methods=['GET'])
def get_training_videos():
    """获取训练视频文件列表API - 使用backend模块"""
    return api_get_training_videos()

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """获取可用模型列表API - 使用backend模块"""
    return api_get_available_models()

@app.route('/api/model-details/<model_type>/<model_name>', methods=['GET'])
def get_model_details(model_type, model_name):
    """获取模型详细信息API - 使用backend模块"""
    return api_get_model_details(model_type, model_name)

@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    """人机对话页面路由 - 处理实时语音交互与智能响应"""

    if request.method == 'POST':
        # ==================== POST请求：处理对话生成 ====================
        try:
            # 收集对话参数
            # ToDo : 这里参数有问题，后续需要调整
            data = {
                "model_name": request.form.get('model_name', 'SyncTalk'),        # 对话模型选择
                "model_param": request.form.get('model_param', ''),              # 模型参数路径
                "voice_clone": request.form.get('voice_clone', 'false'),          # 是否启用语音克隆
                "api_choice": request.form.get('api_choice', 'glm-4-plus')        # API模型选择
            }

            # 调用后端对话引擎
            result = chat_response(data)

            return jsonify({
                'status': 'success',
                'response': result,
                'message': '对话生成成功'
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    # GET请求：渲染人机对话页面
    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    """音频文件保存API - 处理前端上传的录音文件"""

    # 检查是否有音频文件上传
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    # 保存音频文件到指定路径
    audio_path = 'EchOfU/static/audios/input.wav'
    audio_file.save(audio_path)

    return jsonify({'status': 'success', 'message': '音频保存成功', 'path': audio_path})

@app.route('/api/status')
def system_status():
    """系统状态监控API - 获取CPU、内存、GPU等系统资源信息"""

    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent()

        # 获取内存使用情况
        memory = psutil.virtual_memory()

        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')

        # 获取GPU信息（如果可用）
        gpu_info = []
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
            except Exception as e:
                print(f"获取GPU信息失败: {e}")

        return jsonify({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used,
            'memory_total': memory.total,
            'disk_percent': disk.percent,
            'gpus': gpu_info,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/history/<history_type>')
def get_history(history_type):
    """历史记录查询API - 获取指定类型的操作历史记录"""

    history_file = f'static/history/{history_type}.json'

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                data = json.load(f)
                return jsonify(data)
            except:
                return jsonify([])
    else:
        return jsonify([])

@app.route('/video/<path:filename>')
def serve_video(filename):
    """视频文件服务API - 提供生成的视频文件访问"""

    video_path = os.path.join('static', 'videos', filename)
    if os.path.exists(video_path):
        return send_file(video_path)
    else:
        return jsonify({'status': 'error', 'message': '视频文件不存在'}), 404

# =============================================================================
# 应用启动
# =============================================================================
if __name__ == '__main__':
    """
    启动Flask应用
    - debug=True: 启用调试模式，便于开发
    - port=5001: 使用5001端口（避免与其他服务冲突）
    - host='0.0.0.0': 允许外部访问（不仅限于localhost）
    """
    app.run(debug=True, port=5001, host='0.0.0.0')