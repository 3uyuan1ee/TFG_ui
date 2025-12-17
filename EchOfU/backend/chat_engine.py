import speech_recognition as sr
from zhipuai import ZhipuAI

from .voice_generator import OpenVoiceService


def chat_response(data):
    """
    模拟实时对话系统视频生成逻辑。
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # ToDo : 注意这里，与前段对接好录音的存储路径
    # 语音转文字
    # input_audio = "./static/audios/input.wav"
    input_audio = "./SyncTalk/audio/aud.wav"
    input_text = "./static/text/input.txt"
    audio_to_text(input_audio, input_text)

    # ToDo : 这里考虑一下要不要增加一个配置管理，管理一下api这些

    # 大模型回答
    output_text = "./static/text/output.txt"
    api_key = "31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M"
    model = "glm-4-plus"
    ai_response_text = get_ai_response(input_text, output_text, api_key, model)

    # 读取AI回复文本
    with open(output_text, 'r', encoding='utf-8') as f:
        ai_response_text = f.read().strip()

    print(f"[backend.chat_engine] AI回复文本: {ai_response_text}")

    # ToDo : 增加用户选择语音的选项（前端需要实现）

    ov=OpenVoiceService

    # ToDo : 与前段对接数据格式，注意app.py中传过来的data格式
    specker_id = data['speaker_id']
    # 列出可用的语音
    available_speakers = ov.list_available_speakers()
    print(f"[backend.chat_engine] 可用说话人: {available_speakers}")

    # 如果用户选择了语音
    if specker_id in available_speakers:
        speaker_id = data.get('speaker_id')
        print(f"[backend.chat_engine] 使用说话人: {speaker_id}")
    else:
        print(f"[backend.chat_engine] 说话人无效")
        return None


    # OpenVoice语音合成

    voice_path = ov.generate_speech(ai_response_text, speaker_id)

    print(f"[backend.chat_engine] OpenVoice语音合成完成: {voice_path}")

    # 调用ER-NeRF生成视频 ToDo:EN-NeRF待实现
    video_path = generate_video(voice_path)
    
    # video_path = os.path.join("static", "videos", "chat_response.mp4")

    print(f"[backend.chat_engine] 生成视频路径：{video_path}")

    return video_path

def audio_to_text(input_audio, input_text):
    try:
        # 初始化识别器
        recognizer = sr.Recognizer()
        
        # 加载音频文件
        with sr.AudioFile(input_audio) as source:
            # 调整环境噪声
            recognizer.adjust_for_ambient_noise(source)
            # 读取音频数据
            audio_data = recognizer.record(source)
            
            print("正在识别语音...")
            
            # 使用Google语音识别
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            
            # 将结果写入文件
            with open(input_text, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"语音识别完成！结果已保存到: {input_text}")
            print(f"识别结果: {text}")
            
            return text
            
    except sr.UnknownValueError:
        print("无法识别音频内容")
    except sr.RequestError as e:
        print(f"语音识别服务错误: {e}")
    except FileNotFoundError:
        print(f"音频文件不存在: {input_audio}")
    except Exception as e:
        print(f"发生错误: {e}")

def get_ai_response(input_text, output_text, api_key, model):
    client = ZhipuAI(api_key = api_key)
    with open(input_text, 'r', encoding='utf-8') as file:
        content = file.read().strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}]
    )
    output = response.choices[0].message.content

    with open(output_text, 'w', encoding='utf-8') as file:
        file.write(output)

    print(f"答复已保存到: {output_text}")
    return output