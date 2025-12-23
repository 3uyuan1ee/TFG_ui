# Todo List

## 运行环境问题
Docker方案详解

当前架构

EchOfU (Flask)
↓ subprocess
python ER-NeRF/main.py [args]

Docker化后架构

EchOfU (Flask)
↓ subprocess
docker run ernf-container [args]
↓
ER-NeRF (在容器内运行)
