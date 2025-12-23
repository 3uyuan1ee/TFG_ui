#!/bin/bash
# ==============================================================================
# ER-NeRF Docker 容器启动脚本
#
# 支持模式:
#   server          - 启动服务器，等待任务
#   preprocess      - 数据预处理
#   train           - 模型训练
#   test            - 模型测试/推理
#   extract-features - 提取音频特征
#   convert-bfm     - BFM模型转换
#   shell           - 进入交互式shell
# ==============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_code() { echo -e "${CYAN}[CODE]${NC} $1"; }

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate ernerf

# 显示环境信息
show_env() {
    log_info "=================================="
    log_info "ER-NeRF Docker Environment"
    log_info "=================================="
    log_info "Python: $(python --version)"
    log_info "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    log_info "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
    log_info "GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    log_info "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        log_info "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    log_info "Working Directory: $(pwd)"
    log_info "=================================="
    echo ""
}

# ==============================================================================
# 模式1: 服务器模式（等待任务）
# ==============================================================================
start_server() {
    log_info "Starting ER-NeRF Server..."
    show_env

    log_info "Available commands:"
    log_info "  - preprocess: 数据预处理"
    log_info "  - train: 模型训练"
    log_info "  - test: 模型测试/推理"
    log_info "  - extract-features: 提取音频特征"
    log_info "  - convert-bfm: BFM模型转换"
    log_info "  - shell: 进入交互式shell"
    echo ""

    log_info "Server ready. Waiting for tasks..."
    log_info "Press Ctrl+C to stop"

    # 保持容器运行
    tail -f /dev/null
}

# ==============================================================================
# 模式2: 数据预处理
# ==============================================================================
start_preprocess() {
    local VIDEO_PATH=${1}
    local TASK_ID=${2}

    if [ -z "$VIDEO_PATH" ]; then
        log_error "Usage: preprocess <video_path> [task_id]"
        echo ""
        echo "示例:"
        echo "  docker compose run --rm ernerf preprocess /path/to/video.mp4 my_task"
        echo "  docker compose run --rm ernerf preprocess data/my_video.mp4"
        exit 1
    fi

    # 如果没有提供task_id，从视频文件名提取
    if [ -z "$TASK_ID" ]; then
        TASK_ID=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
        log_info "自动提取task_id: $TASK_ID"
    fi

    log_step "开始数据预处理"
    log_info "视频路径: $VIDEO_PATH"
    log_info "任务ID: $TASK_ID"

    cd /workspace/ER-NeRF

    # 检查BFM模型文件
    if [ ! -f "data_utils/face_tracking/3DMM/shape_para.npy" ]; then
        log_warn "========================================"
        log_warn "BFM模型文件未找到！"
        log_warn "========================================"
        log_warn "请按以下步骤操作："
        log_warn "1. 访问 https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details"
        log_warn "2. 注册并下载BFM模型"
        log_warn "3. 将文件放到: ./bfm_models/ 目录"
        log_warn "4. 运行: docker compose run --rm ernerf convert-bfm"
        log_warn "========================================"
        exit 1
    fi

    # 创建数据目录
    local DATA_DIR="/workspace/data/$TASK_ID"
    mkdir -p "$DATA_DIR"

    # 处理视频路径
    if [ ! -f "$VIDEO_PATH" ]; then
        # 尝试相对路径
        if [ -f "/workspace/$VIDEO_PATH" ]; then
            VIDEO_PATH="/workspace/$VIDEO_PATH"
        else
            log_error "视频文件未找到: $VIDEO_PATH"
            exit 1
        fi
    fi

    # 复制视频到数据目录
    log_info "复制视频到: $DATA_DIR/$TASK_ID.mp4"
    cp "$VIDEO_PATH" "$DATA_DIR/$TASK_ID.mp4"

    log_step "运行数据预处理（可能需要几小时）..."
    log_warn "此过程包含以下步骤："
    log_warn "  1. 提取音频"
    log_warn "  2. 提取音频特征（DeepSpeech）"
    log_warn "  3. 提取图像帧（25fps）"
    log_warn "  4. Face parsing"
    log_warn "  5. 提取背景图"
    log_warn "  6. 提取躯干和GT图像"
    log_warn "  7. 提取人脸landmarks"
    log_warn "  8. 生成眨眼数据（EAR算法）"
    log_warn "  9. Face tracking（相机参数估计）"
    log_warn "  10. 生成transforms.json"
    echo ""

    # 运行预处理
    python data_utils/process.py "$DATA_DIR/$TASK_ID.mp4" --task -1 --asr_model deepspeech

    if [ $? -eq 0 ]; then
        log_info "=========================================="
        log_info "预处理完成！"
        log_info "=========================================="
        log_info "数据保存到: $DATA_DIR"
        log_info ""
        log_info "下一步 - 训练模型："
        log_code "docker compose run --rm ernerf train data/$TASK_ID models/ER-NeRF/$TASK_ID"
    else
        log_error "预处理失败！"
        log_error "请查看日志排查问题"
        exit 1
    fi
}

# ==============================================================================
# 模式3: 模型训练
# ==============================================================================
start_train() {
    local DATA_DIR=${1}
    local WORKSPACE=${2}
    local GPU_ID=${3:-0}
    local STAGE=${4:-auto}

    if [ -z "$DATA_DIR" ] || [ -z "$WORKSPACE" ]; then
        log_error "Usage: train <data_dir> <workspace> [gpu_id] [stage]"
        echo ""
        echo "参数说明:"
        echo "  data_dir   - 数据集目录 (例如: data/obama)"
        echo "  workspace  - 模型保存目录 (例如: models/ER-NeRF/obama)"
        echo "  gpu_id     - GPU编号 (默认: 0)"
        echo "  stage      - 训练阶段 (auto|head|lips|torso, 默认: auto)"
        echo ""
        echo "示例:"
        echo "  docker compose run --rm ernerf train data/obama models/ER-NeRF/obama"
        echo "  docker compose run --rm ernerf train data/obama models/ER-NeRF/obama 0 auto"
        exit 1
    fi

    log_step "开始ER-NeRF模型训练"
    log_info "数据集: $DATA_DIR"
    log_info "工作空间: $WORKSPACE"
    log_info "GPU: $GPU_ID"
    log_info "阶段: $STAGE"

    cd /workspace/ER-NeRF

    # 检查数据目录
    if [ ! -d "$DATA_DIR" ]; then
        log_error "数据目录不存在: $DATA_DIR"
        exit 1
    fi

    # 检查transforms文件
    if [ ! -f "$DATA_DIR/transforms_train.json" ]; then
        log_error "未找到transforms_train.json，请先运行预处理"
        exit 1
    fi

    # 创建workspace目录
    mkdir -p "$WORKSPACE"

    # 根据stage选择训练命令
    case "$STAGE" in
        head)
            log_step "阶段1: 头部基础训练 (100k iterations)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --iters 100000
            ;;
        lips)
            log_step "阶段2: 嘴唇微调 (125k iterations)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --iters 125000 --finetune_lips --patch_size 32
            ;;
        torso)
            log_step "阶段3: 躯干训练 (200k iterations)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --torso --iters 200000
            ;;
        auto)
            log_step "自动模式：检测训练进度并继续..."
            # 这里需要调用Python脚本来自动判断阶段
            log_info "正在计算当前训练进度..."

            # 简化版本：直接按顺序执行所有阶段
            log_warn "auto模式将依次执行所有阶段..."

            log_step "阶段1: 头部基础训练 (100k iters)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --iters 100000

            log_step "阶段2: 嘴唇微调 (125k iters)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --iters 125000 --finetune_lips --patch_size 32

            log_step "阶段3: 躯干训练 (200k iters)"
            python main.py "$DATA_DIR" --workspace "$WORKSPACE" -O --torso --iters 200000

            log_info "所有训练阶段完成！"
            ;;
        *)
            log_error "未知阶段: $STAGE"
            exit 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        log_info "=========================================="
        log_info "训练完成！"
        log_info "=========================================="
        log_info "模型保存到: $WORKSPACE"
        log_info ""
        log_info "下一步 - 测试/推理："
        log_code "docker compose run --rm ernerf test $DATA_DIR $WORKSPACE audio.npy"
    else
        log_error "训练失败！"
        exit 1
    fi
}

# ==============================================================================
# 模式4: 模型测试/推理
# ==============================================================================
start_test() {
    local DATA_DIR=${1}
    local WORKSPACE=${2}
    local AUDIO_PATH=${3}
    local TORSO=${4:-true}

    if [ -z "$DATA_DIR" ] || [ -z "$WORKSPACE" ]; then
        log_error "Usage: test <data_dir> <workspace> [audio_path] [torso]"
        echo ""
        echo "参数说明:"
        echo "  data_dir   - 数据集目录"
        echo "  workspace  - 模型目录"
        echo "  audio_path - 音频特征文件 (.npy格式)"
        echo "  torso      - 是否包含躯干 (true|false, 默认: true)"
        echo ""
        echo "示例:"
        echo "  docker compose run --rm ernerf test data/obama models/ER-NeRF/obama audio.npy"
        echo "  docker compose run --rm ernerf test data/obama models/ER-NeRF/obama audio.npy true"
        exit 1
    fi

    log_step "开始ER-NeRF推理"
    log_info "数据集: $DATA_DIR"
    log_info "模型: $WORKSPACE"
    log_info "音频: ${AUDIO_PATH:-无 (使用默认)}"
    log_info "包含躯干: $TORSO"

    cd /workspace/ER-NeRF

    # 构建命令
    CMD="python main.py $DATA_DIR --workspace $WORKSPACE -O --test --test_train"

    if [ "$TORSO" = "true" ]; then
        CMD="$CMD --torso"
    fi

    if [ -n "$AUDIO_PATH" ]; then
        # 处理音频路径
        if [ ! -f "$AUDIO_PATH" ]; then
            if [ -f "/workspace/$AUDIO_PATH" ]; then
                AUDIO_PATH="/workspace/$AUDIO_PATH"
            else
                log_error "音频特征文件未找到: $AUDIO_PATH"
                log_error "请先使用 extract-features 提取音频特征"
                exit 1
            fi
        fi
        CMD="$CMD --aud $AUDIO_PATH"
    fi

    # 添加平滑选项
    CMD="$CMD --smooth_path --smooth_path_window 7"

    log_info "执行命令: $CMD"
    echo ""

    eval $CMD

    if [ $? -eq 0 ]; then
        log_info "=========================================="
        log_info "推理完成！"
        log_info "=========================================="
        log_info "结果保存到: $WORKSPACE/results"
        log_info "或: /workspace/ER-NeRF/results"
    else
        log_error "推理失败！"
        exit 1
    fi
}

# ==============================================================================
# 模式5: 提取音频特征
# ==============================================================================
extract_features() {
    local WAV_PATH=${1}

    if [ -z "$WAV_PATH" ]; then
        log_error "Usage: extract-features <wav_path>"
        echo ""
        echo "示例:"
        echo "  docker compose run --rm ernerf extract-features /path/to/audio.wav"
        echo "  docker compose run --rm ernerf extract-features data/test.wav"
        exit 1
    fi

    log_step "提取DeepSpeech音频特征"
    log_info "音频文件: $WAV_PATH"

    # 处理路径
    if [ ! -f "$WAV_PATH" ]; then
        if [ -f "/workspace/$WAV_PATH" ]; then
            WAV_PATH="/workspace/$WAV_PATH"
        else
            log_error "音频文件未找到: $WAV_PATH"
            exit 1
        fi
    fi

    cd /workspace/ER-NeRF

    log_info "运行DeepSpeech特征提取..."
    python data_utils/deepspeech_features/extract_ds_features.py --input "$WAV_PATH"

    if [ $? -eq 0 ]; then
        OUTPUT_NPY="${WAV_PATH%.wav}.npy"
        if [ -f "$OUTPUT_NPY" ]; then
            log_info "=========================================="
            log_info "特征提取完成！"
            log_info "=========================================="
            log_info "特征保存到: $OUTPUT_NPY"
        else
            log_warn "特征提取成功，但未找到输出文件"
        fi
    else
        log_error "特征提取失败！"
        exit 1
    fi
}

# ==============================================================================
# 模式6: BFM模型转换
# ==============================================================================
convert_bfm() {
    log_step "BFM模型转换"
    echo ""

    cd /workspace/ER-NeRF/data_utils/face_tracking

    # 检查BFM文件
    if [ ! -f "3DMM/bfm_model.npy" ] && [ ! -f "3DMM/BFM_model_front.mat" ]; then
        log_warn "========================================"
        log_warn "未找到BFM模型文件！"
        log_warn "========================================"
        log_warn "请按以下步骤操作："
        log_warn ""
        log_warn "1. 访问 https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details"
        log_warn "2. 注册账号并下载BFM模型"
        log_warn "3. 将文件放置到主机的 ./bfm_models/ 目录"
        log_warn "4. 重新运行此命令"
        log_warn ""
        log_warn "或者直接在容器内放置文件后运行转换："
        log_code "docker compose run --rm ernerf shell"
        log_code "cd /workspace/ER-NeRF/data_utils/face_tracking"
        log_code "python convert_BFM.py"
        log_warn "========================================"
        exit 1
    fi

    log_info "运行BFM转换脚本..."
    python convert_BFM.py

    if [ $? -eq 0 ]; then
        log_info "=========================================="
        log_info "BFM模型转换完成！"
        log_info "=========================================="
        log_info "生成的文件："
        log_info "  - 3DMM/exp_para.npy"
        log_info "  - 3DMM/tex_para.npy"
        log_info "  - 3DMM/shape_para.npy"
        log_info ""
        log_info "现在可以开始数据预处理了！"
    else
        log_error "BFM模型转换失败！"
        exit 1
    fi
}

# ==============================================================================
# 模式7: Shell交互
# ==============================================================================
start_shell() {
    log_info "进入交互式Shell..."
    show_env
    log_info "当前目录: $(pwd)"
    log_info "ER-NeRF目录: /workspace/ER-NeRF"
    echo ""
    exec /bin/bash
}

# ==============================================================================
# 主逻辑
# ==============================================================================
main() {
    log_info "ER-NeRF Docker Container"
    log_info "模式: $1"
    echo ""

    case "$1" in
        server)
            start_server
            ;;
        preprocess)
            start_preprocess "$2" "$3"
            ;;
        train)
            start_train "$2" "$3" "$4" "$5"
            ;;
        test)
            start_test "$2" "$3" "$4" "$5"
            ;;
        extract-features)
            extract_features "$2"
            ;;
        convert-bfm)
            convert_bfm
            ;;
        shell|bash)
            start_shell
            ;;
        *)
            log_error "未知模式: $1"
            echo ""
            echo "可用模式:"
            echo "  server          - 启动服务器，等待任务"
            echo "  preprocess      - 数据预处理"
            echo "  train           - 模型训练"
            echo "  test            - 模型测试/推理"
            echo "  extract-features - 提取音频特征"
            echo "  convert-bfm     - BFM模型转换"
            echo "  shell           - 进入交互式shell"
            echo ""
            echo "使用示例:"
            echo "  # 构建并启动"
            echo "  docker compose -f docker-compose.ernerf.yml build"
            echo "  docker compose -f docker-compose.ernerf.yml up -d"
            echo ""
            echo "  # 预处理"
            echo "  docker compose -f docker-compose.ernerf.yml run --rm ernerf preprocess /path/to/video.mp4 my_task"
            echo ""
            echo "  # 训练"
            echo "  docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/my_task models/ER-NeRF/my_task"
            echo ""
            echo "  # 推理"
            echo "  docker compose -f docker-compose.ernerf.yml run --rm ernerf test data/my_task models/ER-NeRF/my_task audio.npy"
            exit 1
            ;;
    esac
}

main "$@"
