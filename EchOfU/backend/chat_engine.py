import speech_recognition as sr
from zhipuai import ZhipuAI
import os
import shutil
import time
from .path_manager import PathManager
from .voice_generator import get_voice_service,ServiceConfig
# 需要导入 video_generator 中的函数来生成最终视频
from .video_generator import generate_video

def chat_response(data):
    """
    模拟实时对话系统视频生成逻辑。
    流程: 语音转文字 -> LLM回答 -> 文字转语音 -> 语音转视频
    """
    # 初始化路径管理器
    pm = PathManager()
    
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    # 确保输出目录存在
    pm.ensure_directory(pm.get_res_video_path())
    pm.ensure_directory(pm.get_res_voice_path())
    
    # 1. 语音转文字 - 使用path_manager管理路径
    input_audio_dir = pm.get_static_path("audios")
    pm.ensure_directory(input_audio_dir)
    input_audio = os.path.join(input_audio_dir, "input.wav")

    # 文本文件存放目录
    text_dir = pm.get_static_path("text")
    pm.ensure_directory(text_dir)
    input_text_file = os.path.join(text_dir, "input.txt")

    user_text = audio_to_text(input_audio, input_text_file)
    if not user_text:
        print("[backend.chat_engine] 无法识别语音或文件不存在，使用默认问候")
        user_text = "你好" 

    # 2. 大模型回答
    output_text_file = os.path.join(text_dir, "output.txt")
    api_key = "31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M"
    model = "glm-4-plus"
    
    # 获取AI回复
    ai_response_text = get_ai_response(input_text_file, output_text_file, api_key, model)
    print(f"[backend.chat_engine] AI回复文本: {ai_response_text}")

    # 3. 语音合成 (使用新的voice_generator)
    try:
        # 使用人机对话系统中的参考音频
        ref_audio = data.get('ref_audio', '')

        # 创建服务实例
        config = ServiceConfig(enable_vllm=True)
        service = get_voice_service(config)

        print(f"[backend.chat_engine] 参考音频: {ref_audio}")

        # 使用path_manager处理路径转换
        if ref_audio and not os.path.isabs(ref_audio):
            # 将相对路径转换为绝对路径（相对于项目根目录）
            ref_audio = pm.get_static_path(ref_audio)

        # 生成语音 - voice_generator会生成在res_voices中，这里就不用再管理路径了
        timestamp = int(time.time())
        output_filename = f"chat_resp_{timestamp}.wav"

        result = service.clone_voice(
            text=ai_response_text,
            reference_audio=ref_audio if ref_audio else None,
            speed=1.2,
            output_filename=output_filename
        )

        if result.is_success:
            voice_path = result.audio_path
            print(f"[backend.chat_engine] 语音合成完成: {voice_path}")
        else:
            print(f"[backend.chat_engine] 语音合成失败: {result.error_message}")
            voice_path = None

    except Exception as e:
        print(f"[backend.chat_engine] 语音合成错误: {e}")
        voice_path = None

    # 4. 调用视频生成
    # 构造参数调用 video_generator.generate_video
    # 注意: target_text 留空，因为我们已经生成了 voice_path，直接作为 ref_audio 传入
    # 这样 video_generator 就不会再次调用 TTS，而是直接处理我们生成的音频(包括变调和特征提取)
    
    # 确定模型路径
    # 尝试获取前端传来的 model_param，如果没有则使用默认
    # 假设默认有一个名为 'default' 的 ER-NeRF 模型
    default_model_path = pm.get_ernerf_model_path("default")
    model_param = data.get('model_param', default_model_path)

    video_gen_data = {
        'model_name': data.get('model_name', 'SyncTalk'), # 默认模型
        'model_param': model_param,
        'ref_audio': voice_path, # 传入刚才生成的语音
        'gpu_choice': 'GPU0',
        'target_text': '', # 留空，避免 video_generator 重复 TTS
        'pitch': data.get('pitch', 0) # 传递可能的变调参数
    }
    
    # 调用视频生成
    final_video_path = generate_video(video_gen_data)

    print(f"[backend.chat_engine] 生成视频路径：{final_video_path}")
    return final_video_path

def audio_to_text(input_audio, input_text_file):
    """
    使用 SpeechRecognition 将音频转换为文本
    """
    try:
        # 如果文件不存在，直接返回空
        if not os.path.exists(input_audio):
            print(f"[backend.chat_engine] 音频文件不存在: {input_audio}")
            return ""

        recognizer = sr.Recognizer()
        with sr.AudioFile(input_audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            print("正在识别语音...")
            # 这里使用 Google API，需确保网络通畅
            # 也可以根据需求换成 whisper 或其他离线引擎
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            
            with open(input_text_file, 'w', encoding='utf-8') as f:
                f.write(text)
                
            return text
            
    except Exception as e:
        print(f"语音识别错误: {e}")
        return ""

def get_ai_response(input_text_file, output_text_file, api_key, model):
    """
    调用智谱AI大模型获取回答
    """
    try:
        client = ZhipuAI(api_key = api_key)
        
        if not os.path.exists(input_text_file):
            return "无法读取输入文本"

        with open(input_text_file, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        if not content:
            return "听不清，请再说一遍。"

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}]
        )
        output = response.choices[0].message.content

        with open(output_text_file, 'w', encoding='utf-8') as file:
            file.write(output)
            
        return output
    except Exception as e:
        print(f"AI回答生成错误: {e}")
        return "AI服务暂时不可用"



