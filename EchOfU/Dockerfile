# ==============================================================================
# EchOfU 完整应用 Docker Image
# 包含: Flask + CosyVoice + ER-NeRF + 对话系统
#
# 基础镜像: 使用ER-NeRF镜像作为基础
# ==============================================================================

ARG ERNRF_IMAGE=ghcr.io/3uyuan1ee/TFG_ui:ernerf-latest
FROM ${ERNERF_IMAGE}

# 元数据
LABEL maintainer="EchOfU Project" \
      description="EchOfU: AI-Powered Voice Cloning and Video Generation System" \
      version="1.0"

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=5001

# 工作目录
WORKDIR /workspace

# ==============================================================================
# 阶段1: 安装Flask应用依赖
# ==============================================================================
COPY requirements.txt /workspace/requirements.txt

# 激活conda环境并安装依赖
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ernerf && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# 阶段2: 复制应用代码
# ==============================================================================
# 复制后端代码
COPY backend/ /workspace/backend/

# 复制模板文件
COPY templates/ /workspace/templates/

# 复制静态文件
COPY static/ /workspace/static/

# 复制主应用文件
COPY app.py /workspace/

# 复制配置文件
COPY docker/ /workspace/docker/

# 复制子模块代码 (如果需要)
# COPY CosyVoice/ /workspace/CosyVoice/
# COPY SyncTalk/ /workspace/SyncTalk/
# COPY MeloTTS/ /workspace/MeloTTS/
# COPY OpenVoice/ /workspace/OpenVoice/

# ==============================================================================
# 阶段3: 创建必要的目录
# ==============================================================================
RUN mkdir -p /workspace/data \
             /workspace/models/ER-NeRF \
             /workspace/models/CosyVoice \
             /workspace/processed \
             /workspace/static/videos/ref_videos \
             /workspace/static/videos/res_videos \
             /workspace/static/voices/ref_voices \
             /workspace/static/voices/res_voices \
             /workspace/logs

# ==============================================================================
# 阶段4: 配置权限
# ==============================================================================
RUN chmod -R 755 /workspace

# ==============================================================================
# 阶段5: 暴露端口
# ==============================================================================
EXPOSE 5001

# ==============================================================================
# 阶段6: 健康检查
# ==============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# ==============================================================================
# 阶段7: 启动脚本
# ==============================================================================
COPY docker/app/scripts/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# 工作目录
WORKDIR /workspace

# 启动命令
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["server"]