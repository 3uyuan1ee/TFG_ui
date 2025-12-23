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
from backend.voice_generator import get_voice_service, ServiceConfig
from backend.file_manager import file_manager
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
app.config['UPLOAD_FOLDER'] = file_manager.path_manager.get_static_path()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为100MB

# FileManager初始化时会自动确保目录存在
print(f"[App] 使用FileManager进行文件管理")
print(f"[App] 参考音频目录: {file_manager.path_manager.get_ref_voice_path()}")
print(f"[App] 结果音频目录: {file_manager.path_manager.get_res_voice_path()}")

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
            ref_audio_path = request.form.get('ref_audio_path', '').strip()      # 参考音频文件路径
            generate_text = request.form.get('generate_text', '').strip()          # 生成文本内容
            output_filename = request.form.get('output_filename', '').strip()        # 输出文件名

            # 验证必要参数
            if not ref_audio_path:
                return jsonify({
                    'status': 'error',
                    'message': '请选择参考音频'
                }), 400

            if not generate_text:
                return jsonify({
                    'status': 'error',
                    'message': '请输入要生成的文本内容'
                }), 400

            # 创建服务实例
            config = ServiceConfig(enable_vllm=True)  # 启用VLLM加速
            service = get_voice_service(config)

            print(f"[音频克隆] 开始语音克隆:")
            print(f"[音频克隆] 参考音频: {ref_audio_path}")
            print(f"[音频克隆] 生成文本: {generate_text}")

            # 将相对路径转换为绝对路径
            if not os.path.isabs(ref_audio_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ref_audio_path = os.path.join(current_dir, ref_audio_path)
                ref_audio_path = os.path.normpath(ref_audio_path)

            # 执行语音克隆
            result = service.clone_voice(
                text=generate_text,
                reference_audio=ref_audio_path,
                speed=1.2,
                output_filename=output_filename if output_filename else None
            )

            if result.is_success:
                # 转换为相对路径供前端使用
                current_dir = os.path.dirname(os.path.abspath(__file__))
                generated_audio_path = result.audio_path

                if generated_audio_path.startswith(current_dir):
                    relative_path = generated_audio_path[len(current_dir):].lstrip('/\\')
                else:
                    # 如果文件在static/voices/res_voices下，保留路径
                    if 'static/voices/res_voices' in generated_audio_path:
                        relative_path = f"static/voices/res_voices{generated_audio_path.split('static/voices/res_voices')[1]}"
                    else:
                        relative_path = os.path.basename(generated_audio_path)

                print(f"[音频克隆] 语音克隆成功: {result.generation_time:.2f}秒")
                print(f"[音频克隆] 路径转换: {generated_audio_path} -> {relative_path}")

                return jsonify({
                    'status': 'success',
                    'message': '语音克隆成功',
                    'cloned_audio_path': relative_path,
                    'generation_time': result.generation_time
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'语音克隆失败: {result.error_message}'
                }), 500

        except Exception as e:
            print(f"[音频克隆] 请求失败: {e}")
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
    """获取已克隆的音频列表API - 获取生成的音频文件列表"""
    try:
        # 获取生成的音频文件列表（res_voices）
        res_voices_dir = file_manager.path_manager.get_res_voice_path()
        cloned_audios = []

        if os.path.exists(res_voices_dir):
            for filename in os.listdir(res_voices_dir):
                if filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                    file_path = os.path.join(res_voices_dir, filename)
                    file_stat = os.stat(file_path)

                    # 获取相对路径
                    relative_path = file_manager._get_relative_path(file_path)

                    cloned_audios.append({
                        "id": filename,
                        "name": filename,
                        "path": relative_path,
                        "created_at": datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "status": "已生成"
                    })

        # 按创建时间降序排列
        cloned_audios.sort(key=lambda x: x['created_at'], reverse=True)

        print(f"[API] 获取到 {len(cloned_audios)} 个生成的音频")
        return jsonify({
            'status': 'success',
            'audios': cloned_audios,
            'total_count': len(cloned_audios)
        })

    except Exception as e:
        print(f"[API] 获取生成音频列表失败: {e}")
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
            data = {
                "model_name": request.form.get('model_name', 'SyncTalk'),        # 对话模型选择
                "model_param": request.form.get('model_param', ''),              # 模型参数路径
                "ref_audio": request.form.get('ref_audio', ''),                  # 参考音频路径
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
    # 注意：下载大模型时建议关闭debug模式，避免自动重载中断下载
    app.run(debug=False, port=5001, host='0.0.0.0')