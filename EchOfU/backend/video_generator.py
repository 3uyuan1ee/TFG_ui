import os
import time
import subprocess
import shutil
import librosa
import soundfile as sf
from path_manager import PathManager
from voice_generator import OpenVoiceService

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    负责处理音频生成（TTS）、音频变调处理以及调用视频生成模型（SyncTalk/ER-NeRF）。
    """
    # 初始化路径管理器
    pm = PathManager()
    
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # 1. 统一路径配置 - 使用 PathManager
    # 确保输出目录存在
    res_voices_dir = pm.ensure_directory(pm.get_res_voice_path())
    res_videos_dir = pm.ensure_directory(pm.get_res_video_path())

    # 实例化 OpenVoice 服务
    try:
        ov = OpenVoiceService()
    except Exception as e:
        print(f"[backend.video_generator] OpenVoice服务初始化警告: {e}")
        ov = None

    # 获取基础参数
    ref_audio_path = data.get('ref_audio') # 前端传入的参考音频路径
    text = data.get('target_text')         # 要生成的文本 
    speaker_id = data.get('speaker_id', 'default')
    
    # 当前处理的音频路径 (初始为参考音频)
    current_audio_path = ref_audio_path

    
    # 2. 语音生成逻辑 (Text -> Audio)
    # 如果存在 target_text，则忽略 ref_audio，优先使用文本生成新的语音
    if text and text.strip():
        print(f"[backend.video_generator] 检测到目标文本，正在生成语音: {text}")
        if ov:
            # 生成语音 (返回临时路径)
            generated_temp_path = ov.generate_speech(text, speaker_id)
            
            if generated_temp_path and os.path.exists(generated_temp_path):
                # 移动到统一的 res_voices 目录
                timestamp = int(time.time())
                filename = f"tts_{speaker_id}_{timestamp}.wav"
                target_audio_path = pm.get_res_voice_path(filename)
                
                shutil.move(generated_temp_path, target_audio_path)
                current_audio_path = target_audio_path
                print(f"[backend.video_generator] 语音生成成功，已保存至: {current_audio_path}")
            else:
                print("[backend.video_generator] 语音生成失败，将尝试使用原始参考音频")
        else:
            print("[backend.video_generator] OpenVoice服务不可用，跳过语音生成")

    
    # 3. [加分项] 音频变调处理 (Pitch Shift)
    pitch_steps = data.get('pitch')
    if pitch_steps and current_audio_path and os.path.exists(current_audio_path):
        try:
            pitch_steps = float(pitch_steps)
            if pitch_steps != 0:
                print(f"[backend.video_generator] 正在进行音频变调处理: {pitch_steps} steps")
                
                # 加载音频 (保留原始采样率)
                y, sr = librosa.load(current_audio_path, sr=None)
                
                # 变调
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
                
                # 保存变调后的文件到 res_voices
                base_name = os.path.splitext(os.path.basename(current_audio_path))[0]
                shifted_filename = f"{base_name}_pitch_{pitch_steps}.wav"
                shifted_path = pm.get_res_voice_path(shifted_filename)
                
                sf.write(shifted_path, y_shifted, sr)
                
                # 更新当前音频路径指向变调后的文件
                current_audio_path = shifted_path
                print(f"[backend.video_generator] 变调处理完成: {current_audio_path}")
                
        except Exception as e:
            print(f"[backend.video_generator] 音频变调处理失败: {e}")
            # 失败时不中断流程，继续使用变调前的音频

    # 更新 data 中的音频路径，确保后续模型使用最终处理过的音频
    data['ref_audio'] = current_audio_path

    
    # 4. 视频生成模型推理 (SyncTalk / ER-NeRF)
    if not current_audio_path or not os.path.exists(current_audio_path):
        print("[backend.video_generator] 错误: 没有有效的音频输入，无法生成视频")
        return pm.get_res_video_path("error.mp4")

    if data['model_name'] == "SyncTalk":
        try:
            print("[backend.video_generator] 开始 SyncTalk 推理...")
            # 构建 SyncTalk 推理命令
            gpu_id = str(data.get('gpu_choice', '0')).replace("GPU", "")
            
            # SyncTalk 脚本路径
            synctalk_script = pm.get_root_begin_path("SyncTalk", "run_synctalk.sh")
            
            cmd = [
                synctalk_script, 'infer',
                '--model_dir', data['model_param'],
                '--audio_path', current_audio_path,
                '--gpu', gpu_id
            ]

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[backend.video_generator] SyncTalk 警告 (returncode {result.returncode}):")
                print(result.stderr)

            # 结果文件处理 (SyncTalk 逻辑)
            model_dir_name = os.path.basename(data['model_param'])
            source_path = pm.get_root_begin_path("SyncTalk", "model", model_dir_name, "results", "test_audio.mp4")
            
            # 目标文件名
            audio_name = os.path.splitext(os.path.basename(current_audio_path))[0]
            video_filename = f"synctalk_{model_dir_name}_{audio_name}.mp4"
            destination_path = pm.get_res_video_path(video_filename)

            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[backend.video_generator] SyncTalk 视频生成完成: {destination_path}")
                return destination_path
            else:
                # 尝试路径 2: 查找 results 目录下最新的 mp4
                results_dir = pm.get_root_begin_path("SyncTalk", "model", model_dir_name, "results")
                if os.path.exists(results_dir):
                    mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                    if mp4_files:
                        latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                        source_path = os.path.join(results_dir, latest_file)
                        shutil.copy(source_path, destination_path)
                        print(f"[backend.video_generator] 找到最新视频文件: {destination_path}")
                        return destination_path

            print(f"[backend.video_generator] SyncTalk 未找到结果文件: {source_path}")
            return pm.get_res_video_path("out.mp4")

        except Exception as e:
            print(f"[backend.video_generator] SyncTalk 执行异常: {e}")
            return pm.get_res_video_path("error.mp4")

    elif data['model_name'] == "ER-NeRF":
        try:
            print("[backend.video_generator] 开始 ER-NeRF 推理...")
            
            er_nerf_root = pm.get_root_begin_path("ER-NeRF")
            
            # 解析 workspace 名称
            # data['model_param'] 应该是模型的完整路径 "models/ER-NeRF/task_id"
            # 我们只需要 task_id 作为 workspace 参数
            model_path = data.get('model_param', '')
            workspace_name = os.path.basename(model_path.rstrip('/\\'))
            
            # 数据集路径 (推理时需要读取 info.json 等元数据)
            # 对应训练时的 preprocess_data_path: models/ER-NeRF/data/<task_id>
            dataset_path = pm.get_ernerf_data_path(workspace_name)
            
            if not workspace_name:
                print("[backend.video_generator] 错误: 无法获取 workspace 名称")
                return pm.get_res_video_path("error.mp4")

            # 构建 ER-NeRF 推理命令
            cmd = [
                "python", os.path.join(er_nerf_root, "main.py"),
                dataset_path,
                "--workspace", model_path, # 使用完整路径
                "--aud", current_audio_path,
                "--test",       # 推理模式
                "--test_train"  # 使用训练集视角
            ]
            
            # GPU 设置
            env = os.environ.copy()
            if 'gpu_choice' in data:
                gpu_id = str(data['gpu_choice']).replace("GPU", "")
                env['CUDA_VISIBLE_DEVICES'] = gpu_id

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")
            
            # 执行推理 (切换 cwd 到 ER-NeRF 目录以避免相对路径问题)
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=er_nerf_root 
            )

            # 结果文件处理
            # 搜索结果策略
            possible_result_dirs = [
                os.path.join(model_path, "results"),                  # models/ER-NeRF/id/results
                os.path.join(er_nerf_root, "results", workspace_name), # ER-NeRF/results/id
                os.path.join(er_nerf_root, workspace_name, "results")  # ER-NeRF/id/results
            ]
            
            timestamp = int(time.time())
            video_filename = f"ernerf_{workspace_name}_{timestamp}.mp4"
            destination_path = pm.get_res_video_path(video_filename)
            
            found_video = False
            for results_dir in possible_result_dirs:
                if os.path.exists(results_dir):
                    # 查找最新的 mp4
                    mp4_files = []
                    for root, dirs, files in os.walk(results_dir):
                        for f in files:
                            if f.endswith('.mp4'):
                                mp4_files.append(os.path.join(root, f))
                    
                    if mp4_files:
                        latest_video = max(mp4_files, key=os.path.getctime)
                        print(f"[backend.video_generator] 找到视频: {latest_video}")
                        shutil.copy(latest_video, destination_path)
                        found_video = True
                        break
            
            if found_video:
                print(f"[backend.video_generator] ER-NeRF 视频生成成功: {destination_path}")
                return destination_path
            else:
                print("[backend.video_generator] ER-NeRF 推理完成但未找到生成的视频文件")
                return pm.get_res_video_path("out.mp4")

        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] ER-NeRF 命令执行失败 (code {e.returncode})")
            print(f"Stderr: {e.stderr}")
            return pm.get_res_video_path("error.mp4")
        except Exception as e:
            print(f"[backend.video_generator] ER-NeRF 其他错误: {e}")
            return pm.get_res_video_path("error.mp4")
    
    # 默认返回
    default_path = pm.get_res_video_path("out.mp4")
    print(f"[backend.video_generator] 未匹配模型或发生错误，返回默认路径: {default_path}")
    return default_path
