# API接口数据格式规范

## 1. 视频生成接口 (/video_generation)

**请求方式**: POST
**Content-Type**: multipart/form-data

**请求数据格式 (FormData)**
```javascript
const formData = {
    model_name: "SyncTalk",           // 必填 - 模型名称
    model_param: "/path/to/model",    // 必填 - 模型目录地址
    ref_audio_id: "speaker_1",        // 必填 - 参考音频ID (注意：不是路径)
    gpu_choice: "GPU0",               // 必填 - GPU选择
    target_text: "要生成的文本"         // 必填
}
```

**字段说明**

| 字段名         | 类型     | 必填  | 可选值                    | 说明                                     |
|---------------|--------|-----|-------------------------|----------------------------------------|
| model_name    | string | ✅   | SyncTalk, ER-NeRF       | 选择要使用的AI模型，SyncTalk需要原音频，ER-NeRF只需要文本 |
| model_param   | string | ✅   | 任意有效路径                | 模型文件存储的目录地址                           |
| ref_audio_id  | string | ✅   | 已注册的说话人ID            | 参考音频的ID，从克隆音频列表获取                     |
| gpu_choice    | string | ✅   | GPU0, GPU1              | 选择使用的GPU设备                            |
| target_text   | string | ⚠️   | 任意文本                   | ER-NeRF模式必填，SyncTalk模式可选（为空则使用参考音频原文） |

**响应数据格式**
```json
{
    "status": "success",                    // 处理状态: "success" | "error"
    "video_path": "/static/videos/generated_video.mp4",  // 生成视频的访问路径
    "message": "视频生成成功"                  // 可选 - 错误或成功信息
}
```

---

## 2. 模型训练接口 (/model_training)

**请求方式**: POST
**Content-Type**: multipart/form-data

**请求数据格式 (FormData)**
```javascript
const formData = {
    model_choice: "SyncTalk",         // 必填 - 训练模型选择
    ref_video: "/path/to/video.mp4",  // 必填 - 参考视频地址
    gpu_choice: "GPU0",               // 必填 - GPU选择
    epoch: "100",                     // 必填 - 训练轮数
    custom_params: "lr=0.001",        // 可填 - 自定义训练参数
}
```

**字段说明**

| 字段名           | 类型     | 必填  | 可选值                    | 说明                    |
|---------------|--------|-----|-------------------------|-----------------------|
| model_choice  | string | ✅   | SyncTalk, ER-NeRF       | 选择要训练的模型类型           |
| ref_video     | string | ✅   | 任意视频文件路径              | 参考视频文件的路径            |
| gpu_choice    | string | ✅   | GPU0, GPU1              | 选择训练使用的GPU           |
| epoch         | number | ✅   | 正整数(1-∞)               | 训练轮数，默认值为100         |
| custom_params | string | ❌   | 训练参数字符串               | JSON格式的自定义训练参数      |
| speaker_id    | string | 🔍   | 自动生成                   | 系统自动生成的说话人ID，格式为user_YYYYMMDD |

**响应数据格式**
```json
{
    "status": "success",              // 处理状态: "success" | "error"
    "task_id": "train_20251218_054414",  // 训练任务ID，用于进度查询
    "message": "模型训练开始"              // 操作状态信息
}
```

---

## 3. 音频克隆接口 (/audio_clone)

**请求方式**: POST
**Content-Type**: multipart/form-data

**支持两种模式：克隆特征提取模式 和 音频生成模式**

### 3.1 克隆特征提取模式
```javascript
const formData = {
    original_audio_path: "/path/to/audio.wav",  // 必填 - 原始音频路径
    audio_id: "new_speaker_1",                  // 必填 - 新生成的说话人ID
    target_audio_id: "",                        // 克隆模式留空
    gen_audio_id: "",                           // 克隆模式留空
    generate_text: ""                           // 克隆模式留空
}
```

### 3.2 音频生成模式
```javascript
const formData = {
    original_audio_path: "",                    // 生成模式留空
    audio_id: "existing_speaker_1",             // 必填 - 已存在的说话人ID
    target_audio_id: "",                        // 生成模式留空
    gen_audio_id: "generated_audio_1",          // 必填 - 生成音频的ID
    generate_text: "要生成的文本内容"              // 必填 - 要合成的文本
}
```

**字段说明**

| 字段名                 | 类型     | 必填  | 说明                          |
|----------------------|--------|-----|-----------------------------|
| original_audio_path  | string | ⚠️   | 克隆模式必填，生成模式留空                |
| audio_id             | string | ✅   | 克隆模式为新建ID，生成模式为已存在ID         |
| target_audio_id      | string | ❌   | 预留字段，当前版本留空                 |
| gen_audio_id         | string | ⚠️   | 生成模式必填，克隆模式留空                |
| generate_text        | string | ⚠️   | 生成模式必填，克隆模式留空                |

**响应数据格式**

**克隆模式响应**:
```json
{
    "status": "success",
    "message": "音频特征提取成功",
    "speaker_id": "new_speaker_1"
}
```

**生成模式响应**:
```json
{
    "status": "success",
    "message": "音频生成成功",
    "cloned_audio_path": "/static/voices/generated_audio.wav"
}
```

---

## 4. 人机对话系统接口 (/chat_system)

**请求方式**: POST
**Content-Type**: multipart/form-data

**请求数据格式 (FormData)**
```javascript
const formData = {
    audio: audioBlob,                    // 必填 - 用户录音文件
    model_name: "SyncTalk",              // 必填 - 对话模型名称
    model_param: "/path/to/model",       // 必填 - 模型参数路径
    ref_audio_id: "speaker_1",           // 必填 - 回应音频的说话人ID
    api_choice: "glm-4-plus"             // 必填 - 对话API选择
}
```

**字段说明**

| 字段名         | 类型     | 必填  | 可选值                    | 说明                    |
|---------------|--------|-----|-------------------------|-----------------------|
| audio         | Blob   | ✅   | 录音音频文件                 | 用户录制的语音文件             |
| model_name    | string | ✅   | SyncTalk                | 用于生成回应视频的模型          |
| model_param   | string | ✅   | 任意有效路径                | 模型文件存储的目录地址           |
| ref_audio_id  | string | ✅   | 已注册的说话人ID            | AI回应使用的说话人音色         |
| api_choice    | string | ✅   | glm-4-plus              | 对话使用的语言模型API         |

**响应数据格式**
```json
{
    "status": "success",              // 处理状态: "success" | "error"
    "response": "AI回复内容",            // AI生成的文本回复
    "message": "对话生成成功",              // 操作状态信息
    "video_path": "/static/videos/chat_response.mp4"  // 回应视频路径
}
```

---

## 5. 音频文件保存接口 (/save_audio)

**请求方式**: POST
**Content-Type**: multipart/form-data

**请求数据格式 (FormData)**
```javascript
const formData = {
    audio: audioBlob,                    // 必填 - 录音音频文件
}
```

**字段说明**

| 字段名   | 类型   | 必填  | 说明                       |
|-------|------|-----|--------------------------|
| audio | Blob | ✅   | 用户录制的音频文件，自动命名为input.wav |

**响应数据格式**
```json
{
    "status": "success",              // 处理状态: "success" | "error"
    "message": "音频保存成功"         // 操作结果信息
}
```

---

## 6. 系统状态监控接口 (/api/status)

**请求方式**: GET
**Content-Type**: application/json

**响应数据格式**
```json
{
    "cpu_percent": 45.2,              // CPU使用率百分比
    "memory_percent": 67.8,           // 内存使用率百分比
    "memory_used": 8589934592,        // 已使用内存(字节)
    "memory_total": 12782643200,      // 总内存(字节)
    "disk_percent": 78.5,             // 磁盘使用率百分比
    "gpus": [                         // GPU状态数组
        {
            "name": "NVIDIA GeForce RTX 3090",
            "load": 65.3,             // GPU负载百分比
            "memory_used": 8589934592,  // GPU已使用内存
            "memory_total": 24297080832, // GPU总内存
            "temperature": 72         // GPU温度(摄氏度)
        }
    ],
    "timestamp": "2025-12-18T05:44:14.123Z"  // 时间戳
}
```

---

## 7. 已克隆音频列表接口 (/api/cloned-audios)

**请求方式**: GET
**Content-Type**: application/json

**响应数据格式**
```json
{
    "status": "success",              // 处理状态: "success" | "error"
    "audios": [                       // 音频列表数组
        {
            "id": "speaker_1",        // 说话人ID
            "name": "speaker_1",      // 显示名称
            "created_at": "2025-12-18 05:44:14",  // 创建时间
            "reference_audio": "unknown",         // 参考音频路径
            "status": "已提取特征"       // 状态描述
        }
    ],
    "total_count": 1                  // 总数量
}
```

---

## 8. 历史记录查询接口 (/api/history/<history_type>)

**请求方式**: GET
**Content-Type**: application/json

**路径参数**
- `history_type`: 历史记录类型，支持以下值：
  - `video_generation` - 视频生成历史
  - `model_training` - 模型训练历史
  - `audio_clone` - 音频克隆历史
  - `chat_system` - 人机对话历史

**响应数据格式**
```json
{
    "status": "success",              // 处理状态: "success" | "error"
    "history": [                      // 历史记录数组
        {
            "id": "record_1",         // 记录ID
            "timestamp": "2025-12-18 05:44:14",  // 操作时间
            "parameters": {},         // 操作参数
            "result": {},             // 操作结果
            "status": "completed"     // 操作状态
        }
    ],
    "total_count": 1                  // 总记录数
}
```

---

## 9. 视频文件服务接口 (/video/<path:filename>)

**请求方式**: GET
**Content-Type**: video/mp4

**路径参数**
- `filename`: 视频文件名

**功能**: 提供生成的视频文件访问服务

---

## 🔧 前端JavaScript处理逻辑

### 统一的API调用模式

```javascript
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
        alert('操作失败！' + data.message);
    }
})
.catch(err => console.error('API调用错误:', err));
```

### 视频路径处理

```javascript
// 防止缓存的视频路径拼接
const newSrc = data.video_path + '?t=' + new Date().getTime();

// Windows路径转换为Unix路径
const unixPath = "/" + windowsPath.replace(/\\/g, "/");
```

### 录音功能数据处理

```javascript
// 录音数据处理
const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
const formData = new FormData();
formData.append('audio', audioBlob, 'input.wav');
```

### 错误处理最佳实践

```javascript
try {
    const response = await fetch('/api_endpoint', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.status === 'success') {
        // 处理成功响应
        handleSuccess(data);
    } else {
        // 处理业务错误
        handleError(data.message);
    }
} catch (error) {
    // 处理网络错误或其他异常
    console.error('API调用失败:', error);
    handleError('网络连接失败，请检查网络设置');
}
```

### 进度状态管理

```javascript
// 显示加载状态
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    element.innerHTML = '处理中...';
    element.disabled = true;
}

// 隐藏加载状态
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    element.disabled = false;
}

// 模拟进度条更新
function updateProgressBar(progress, message) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    progressBar.style.width = `${progress}%`;
    progressText.textContent = message;
}
```

## 📝 注意事项

1. **参数验证**: 前端应进行基本参数验证，但最终验证由后端处理
2. **错误处理**: 所有API调用都应包含完善的错误处理逻辑
3. **文件上传**: 大文件上传时应显示进度条并提供取消功能
4. **路径处理**: 注意Windows和Linux路径格式的差异
5. **缓存控制**: 视频文件应添加时间戳参数避免浏览器缓存
6. **异步处理**: 长时间运行的任务应提供状态查询机制
7. **安全性**: 用户上传的文件应进行类型和大小验证
8. **兼容性**: 确保API调用在不同浏览器中的兼容性