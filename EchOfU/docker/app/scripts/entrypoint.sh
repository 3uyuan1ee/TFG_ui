#!/bin/bash
# ==============================================================================
# EchOfU Flask应用启动脚本
# ==============================================================================

set -e

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate ernerf

# 进入工作目录
cd /workspace

echo "========================================"
echo "EchOfU Flask Application"
echo "========================================"
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$TORCH_CUDA_ARCH_LIST" != "" ]; then
    echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi
echo "========================================"
echo ""

# 根据传入的参数执行不同的操作
case "$1" in
  server)
    echo "启动Flask服务器..."
    echo "端口: ${PORT:-5001}"
    echo ""

    # 启动Flask应用
    exec python app.py
    ;;

  shell)
    echo "进入交互式Shell..."
    exec /bin/bash
    ;;

  test)
    echo "运行测试..."
    exec pytest -v
    ;;

  *)
    # 执行自定义命令
    exec "$@"
    ;;
esac