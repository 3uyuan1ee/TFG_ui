API接口数据格式规范

1. 视频生成接口 (/video_generation)

请求方式: POSTContent-Type: multipart/form-data

请求数据格式 (FormData)

const formData = {
model_name: "SyncTalk",           // 必填 - 模型名称
model_param: "/path/to/model",    // 必填 - 模型目录地址
ref_audio: "/path/to/audio.wav",  // 必填 - 参考音频地址  
gpu_choice: "GPU0",               // 必填 - GPU选择
target_text: "要生成的文本"        // 可填 - 目标文字，留空则使用参考音频
}

字段说明

| 字段名         | 类型     | 必填  | 可选值                      | 说明                 |
  |-------------|--------|-----|--------------------------|--------------------|
| model_name  | string | ✅   | model1, model2, SyncTalk | 选择要使用的AI模型         |
| model_param | string | ✅   | 任意有效路径                   | 模型文件存储的目录地址        |
| ref_audio   | string | ✅   | 任意音频文件路径                 | 参考音频文件的相对或绝对路径     |
| gpu_choice  | string | ✅   | GPU0, GPU1               | 选择使用的GPU设备         |
| target_text | string | ❌   | 任意文本                     | 要合成的文本内容，为空则使用音频原文 |

响应数据格式

{
"status": "success",              // 处理状态: "success" | "error"
"video_path": "/static/videos/generated_video.mp4",  // 生成视频的访问路径
"message": "视频生成成功"         // 可选 - 错误或成功信息
}

  ---
2. 模型训练接口 (/model_training)

请求方式: POSTContent-Type: multipart/form-data

请求数据格式 (FormData)

const formData = {
model_choice: "SyncTalk",         // 必填 - 训练模型选择
ref_video: "/path/to/video.mp4",  // 必填 - 参考视频/图像地址
gpu_choice: "GPU0",               // 必填 - GPU选择
epoch: "10",                      // 必填 - 训练轮数
custom_params: "lr=0.001",        // 可填 - 自定义训练参数
speaker_id: "user_20251215"       // 可填 - 说话人ID（未在前端显示）
}

字段说明

| 字段名           | 类型     | 必填  | 可选值                      | 说明             |
  |---------------|--------|-----|--------------------------|----------------|
| model_choice  | string | ✅   | modelA, modelB, SyncTalk | 选择要训练的模型类型     |
| ref_video     | string | ✅   | 任意视频/图像路径                | 参考视频或图像文件的路径   |
| gpu_choice    | string | ✅   | GPU0, GPU1               | 选择训练使用的GPU     |
| epoch         | number | ✅   | 正整数(1-∞)                 | 训练轮数，默认值为10    |
| custom_params | string | ❌   | 训练参数字符串                  | JSON格式的自定义训练参数 |

响应数据格式

{
"status": "success",              // 处理状态: "success" | "error"
"video_path": "/static/videos/training_result.mp4",  // 训练结果视频路径
"message": "训练完成"             // 可选 - 状态信息
}

  ---
3. 实时对话系统接口 (/chat_system)

请求方式: POSTContent-Type: multipart/form-data

请求数据格式 (FormData)

const formData = {
model_name: "SyncTalk",           // 必填 - 模型名称
model_param: "/path/to/model",    // 必填 - 模型目录地址
voice_clone: "cloneA",            // 必填 - 语音克隆模型
api_choice: "openai",             // 必填 - 对话API选择
speaker_id: "test_speaker_1"      // 可填 - 指定说话人ID
}

字段说明

| 字段名         | 类型     | 必填  | 可选值                      | 说明         |
  |-------------|--------|-----|--------------------------|------------|
| model_name  | string | ✅   | model1, model2, SyncTalk | 视频生成模型选择   |
| model_param | string | ✅   | 任意有效路径                   | 模型参数文件路径   |
| voice_clone | string | ✅   | cloneA, cloneB           | 语音克隆模型选择   |
| api_choice  | string | ✅   | openai, azure            | 对话API服务选择  |
| speaker_id  | string | ❌   | 已注册的说话人ID                | 指定使用的说话人音色 |

响应数据格式

{
"status": "success",              // 处理状态: "success" | "error"
"video_path": "/static/videos/chat_response.mp4",     // 对话生成的视频路径
"message": "对话完成"             // 可选 - 状态信息
}

  ---
4. 音频保存接口 (/save_audio)

请求方式: POSTContent-Type: multipart/form-data

请求数据格式 (FormData)

const formData = {
audio: Blob,                      // 必填 - 录音音频文件
}

字段说明

| 字段名   | 类型   | 必填  | 说明                       |
  |-------|------|-----|--------------------------|
| audio | Blob | ✅   | 用户录制的音频文件，自动命名为input.wav |

响应数据格式

{
"status": "success",              // 处理状态: "success" | "error"
"message": "音频保存成功"         // 操作结果信息
}

  ---
🔧 前端JavaScript处理逻辑

统一的API调用模式

fetch('/api_endpoint', {
method: 'POST',
body: formData
})
.then(res => res.json())
.then(data => {
console.log("后端返回:", data);
if (data.status === 'success') {
// 成功处理逻辑
const videoEl = document.getElementById('outputVideo');
videoEl.src = data.video_path + '?t=' + new Date().getTime();
videoEl.load();
videoEl.play().catch(err => console.warn('自动播放被阻止:', err));
} else {
// 错误处理逻辑
alert('操作失败！');
}
})
.catch(err => console.error('API调用错误:', err));

视频路径处理

// 防止缓存的视频路径拼接
const newSrc = data.video_path + '?t=' + new Date().getTime();

// Windows路径转换为Unix路径
const unixPath = "/" + windowsPath.replace("\\", "/");

录音功能数据处理

// 录音数据处理
const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
const formData = new FormData();
formData.append('audio', audioBlob, 'input.wav');
