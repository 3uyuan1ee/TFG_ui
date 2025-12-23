# API 文档

说明：本文件列出前端与后端对接使用的 HTTP 接口、请求字段（data）格式、示例请求类型与期望返回值。

总体约定：
- 文件上传使用 `multipart/form-data`（文件字段在 `request.files`）。
- 表单参数多为 `application/x-www-form-urlencoded`（字段在 `request.form`）。
- 成功通常返回 JSON，包含 `status: 'success'`；出错返回 `status: 'error'` 并可能伴随 HTTP 错误码（400/404/500）。

---

## 1. 上传参考音频
- URL: `/api/upload-reference-audio`
- Method: `POST`
- Content-Type: `multipart/form-data`
- 请求字段（form-data）:
  - `audio` (file) — 支持扩展名：`.wav, .mp3, .m4a, .flac, .ogg`
- 成功响应 (200):

```json
{
  "status": "success",
  "message": "音频上传成功",
  "filename": "safe_filename.wav",
  "relative_path": "static/voices/ref_voices/safe_filename.wav",
  "original_name": "原始文件名.wav"
}
```

- 常见错误：
  - 400: 没有文件或格式/大小不合格 -> {"status":"error","message":"..."}
  - 500: 服务器异常 -> {"status":"error","message":"音频上传失败: ..."}

---

## 2. 获取参考音频列表
- URL: `/api/reference-audios`
- Method: `GET`
- 请求 data: 无
- 成功响应 (200):

```json
{
  "status": "success",
  "files": [
    {
      "filename": "xxx.wav",
      "size": 12345,
      "size_mb": 0.01,
      "created_time": "2025-01-01 12:00:00",
      "modified_time": "2025-01-01 12:00:00",
      "file_type": "WAV",
      "relative_path": "static/voices/ref_voices/xxx.wav"
    }
  ],
  "total_count": 1
}
```

- 错误：500 返回 {"status":"error","message":"...","files":[],"total_count":0}

---

## 3. 上传训练视频
- URL: `/api/upload-training-video`
- Method: `POST`
- Content-Type: `multipart/form-data`
- 请求字段（form-data）:
  - `video` (file) — 支持扩展名：`.mp4, .avi, .mov, .mkv, .flv, .webm`。
  - 后端 `FileManager` 验证默认最大约 200MB（可在代码中调整）。
- 成功响应 (200):

```json
{
  "status": "success",
  "message": "视频上传成功",
  "filename": "safe_name.mp4",
  "relative_path": "static/videos/ref_videos/safe_name.mp4",
  "original_name": "原始文件名.mp4"
}
```

- 错误：400/500，返回 {"status":"error","message":"..."}

---

## 4. 获取训练视频列表
- URL: `/api/training-videos`
- Method: `GET`
- 请求 data: 无
- 成功响应 (200):

```json
{
  "status": "success",
  "files": [ /* file info 与参考音频类似，包含 relative_path */ ],
  "total_count": 3
}
```

---

## 5. 获取可用模型列表
- URL: `/api/available-models`
- Method: `GET`
- 请求 data: 无
- 成功响应 (200):

```json
{
  "status": "success",
  "models": [
    {
      "name": "task_id",
      "type": "ER-NeRF",
      "path": "models/ER-NeRF/task_id",
      "model_files_count": 5,
      "created_time": "2025-...",
      "description": "ER-NeRF 模型 - 包含 5 个模型文件"
    }
  ],
  "total_count": 1
}
```

---

## 6. 获取模型详情
- URL: `/api/model-details/<model_type>/<model_name>`
- Method: `GET`
- Path 参数:
  - `model_type` (string) — 示例：`ER-NeRF` 或 `SyncTalk`
  - `model_name` (string) — 模型目录名或 `root`
- 成功响应 (200):

```json
{
  "status": "success",
  "model_details": {
    "name": "task_id",
    "type": "ER-NeRF",
    "path": "models/ER-NeRF/task_id",
    "total_files": 10,
    "total_size_mb": 123.45,
    "created_time": "...",
    "modified_time": "...",
    "files": [ { "filename","relative_path","size","size_mb","file_type","created_time","modified_time" } ]
  }
}
```

- 错误：
  - 400: 不支持的模型类型 -> HTTP 400 + {"status":"error","message":"..."}
  - 404: 模型目录不存在 -> HTTP 404 + {"status":"error","message":"..."}

---

## 7. 获取已克隆音频（说话人特征列表）
- URL: `/api/cloned-audios`
- Method: `GET`
- 请求 data: 无
- 成功响应 (200):

```json
{
  "status": "success",
  "audios": [
    { "id": "spk1", "name":"spk1", "created_at":"...", "reference_audio":"...","status":"已提取特征" }
  ],
  "total_count": 1
}
```

---

## 8. 保存前端录音
- URL: `/save_audio`
- Method: `POST`
- Content-Type: `multipart/form-data`
- 请求字段：
  - `audio` (file)
- 成功响应 (200):

```json
{ "status": "success", "message": "音频保存成功", "path": "EchOfU/static/audios/input.wav" }
```

---

## 9. 视频生成（页面表单调用）
- URL: `/video_generation`
- Method: `POST`
- Content-Type: `application/x-www-form-urlencoded` 或 `multipart/form-data`
- 请求字段（form）:
  - `model_name` (string) — 如 `SyncTalk` 或 `ER-NeRF`
  - `model_param` (string) — 模型参数路径，例如 `models/ER-NeRF/<id>` 或 SyncTalk 模型目录
  - `ref_audio` (string) — 参考音频相对/绝对路径（若提供 `target_text` 则可忽略）
  - `gpu_choice` (string) — 如 `GPU0`
  - `target_text` (string) — 若提供后端会优先用 TTS 生成语音
  - 可选：`speaker_id`、`pitch`
- 成功响应 (200):

```json
{ "status":"success","video_path":"static/videos/res_videos/xxx.mp4","message":"视频生成成功" }
```

- 失败：返回 {"status":"error","message":"..."} 或返回默认/错误路径（参见后端实现）

---

## 10. 模型训练（页面表单调用）
- URL: `/model_training`
- Method: `POST`
- Content-Type: `application/x-www-form-urlencoded`
- 请求字段（form）:
  - `model_choice` (string) — `SyncTalk` 或 `ER-NeRF`
  - `ref_video` (string) — 参考视频路径
  - `gpu_choice` (string) — `GPU0`
  - `epoch` (int/string) — 训练轮次
  - `custom_params` (string) — 可选，自定义参数，如 `lr=0.001,batch_size=4`
- 成功响应 (200):

```json
{ "status":"success","message":"模型训练开始","task_id":"train_YYYYMMDD_HHMMSS" }
```

- 备注：当前没有实时训练进度接口。如需轮询进度，可新增 `GET /api/train-status/<task_id>`。

---

## 11. 音频克隆 / 生成
- URL: `/audio_clone`
- Method: `POST`
- Content-Type: `application/x-www-form-urlencoded`
- 请求字段（form）:
  - `original_audio_path` (string) — 克隆模式：参考音频（相对或绝对路径）
  - `audio_id` (string) — 生成模式：已有音频 ID（作为 `speaker_id`）
  - `target_audio_id` (string) — 克隆目标 ID（保存特征时使用）
  - `gen_audio_id` (string) — 可选
  - `generate_text` (string) — 若存在则走“生成模式”

- 两种模式返回示例：
  - 生成模式（`generate_text` 非空）成功：

```json
{ "status":"success","message":"音频生成成功","cloned_audio_path":"static/voices/res_voices/xxx.wav" }
```

  - 克隆模式（提取特征）成功：

```json
{ "status":"success","message":"说话人特征提取完成，可以用于后续音频生成" }
```

---

## 12. 人机对话（整合生成视频）
- URL: `/chat_system`
- Method: `POST`
- Content-Type: `application/x-www-form-urlencoded`
- 请求字段（form）:
  - `model_name`, `model_param`, `voice_clone`, `api_choice`，可选 `speaker_id`, `pitch`
- 行为说明：后端会读取 `EchOfU/static/audios/input.wav` 做语音识别 -> LLM -> TTS -> 调用 `generate_video`。
- 返回：

```json
{ "status":"success","response":"/static/videos/res_videos/xxx.mp4","message":"对话生成成功" }
```

---

## 13. 系统状态
- URL: `/api/status`
- Method: `GET`
- 返回：

```json
{
  "cpu_percent": 12.3,
  "memory_percent": 45.6,
  "memory_used": 123456789,
  "memory_total": 987654321,
  "disk_percent": 70.2,
  "gpus": [ { "name":"GPU0","load":50.0,"memory_used":4000,"memory_total":8192,"temperature":60 } ],
  "timestamp": "2025-12-19T12:00:00"
}
```

---

## 14. 历史记录查询
- URL: `/api/history/<history_type>`（例如 `video_generation`）
- Method: `GET`
- 返回：对应 `static/history/<history_type>.json` 内容，文件不存在时返回 `[]`。

---

## 15. 视频文件访问
- URL: `/video/<path:filename>`
- Method: `GET`
- 返回：视频文件流或 404 JSON 错误：{"status":"error","message":"视频文件不存在"}

---

## 对接注意事项（简要）
- 文件字段名必须与后端一致：音频 `audio`，视频 `video`。
- 后端返回的路径常为项目内部相对路径（`static/...` 或 `models/...`），前端使用时请拼接服务地址或通过后端的静态访问路由访问。
- 训练和生成类操作可能耗时，建议前端以异步方式调用并提供用户提示或轮询机制。
- 如需新增 API（例如训练进度/日志查询），建议统一返回 `status/message/data` 格式并增加 `task_id` 支持。

---