#!/bin/bash
# ==============================================================================
# ER-NeRF Docker 快速启动脚本
# ==============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 检查Docker
check_docker() {
    log_step "检查Docker环境..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker未运行，请启动Docker"
        exit 1
    fi

    log_info "Docker环境正常"
}

# 检查NVIDIA Docker
check_nvidia_docker() {
    log_step "检查NVIDIA Docker支持..."

    if ! docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu18.04 nvidia-smi &> /dev/null; then
        log_warn "NVIDIA Docker支持未正确配置"
        log_warn "请安装nvidia-container-toolkit"
        log_warn "参考: https://github.com/NVIDIA/nvidia-docker"
        exit 1
    fi

    log_info "NVIDIA Docker支持正常"
}

# 检查BFM模型
check_bfm() {
    log_step "检查BFM模型..."

    if [ ! -d "./bfm_models" ]; then
        log_warn "未找到bfm_models目录"
        log_info "创建目录..."
        mkdir -p ./bfm_models
    fi

    # 检查关键文件
    if [ ! -f "./bfm_models/shape_para.npy" ]; then
        log_warn "未找到BFM模型文件"
        echo ""
        log_warn "========================================"
        log_warn "BFM模型未配置！"
        log_warn "========================================"
        echo ""
        log_info "请按以下步骤操作："
        log_info "1. 访问: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details"
        log_info "2. 注册并下载BFM模型"
        log_info "3. 将文件放到 ./bfm_models/ 目录"
        log_info "4. 运行: bash scripts/start_ernerf_docker.sh setup-bfm"
        echo ""
        read -p "是否已经准备好BFM模型? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "请准备好BFM模型后重新运行"
            exit 1
        fi
    else
        log_info "BFM模型文件已找到"
    fi
}

# 构建镜像
build_image() {
    log_step "构建ER-NeRF Docker镜像..."
    log_info "这可能需要30-60分钟，请耐心等待..."

    docker compose -f docker-compose.ernerf.yml build

    if [ $? -eq 0 ]; then
        log_info "镜像构建成功"
    else
        log_error "镜像构建失败"
        exit 1
    fi
}

# 启动容器
start_container() {
    log_step "启动ER-NeRF容器..."

    docker compose -f docker-compose.ernerf.yml up -d

    if [ $? -eq 0 ]; then
        log_info "容器启动成功"
        log_info "查看日志: docker compose -f docker-compose.ernerf.yml logs -f"
    else
        log_error "容器启动失败"
        exit 1
    fi
}

# 转换BFM模型
setup_bfm() {
    log_step "转换BFM模型..."

    if [ ! -f "./bfm_models/shape_para.npy" ]; then
        log_error "BFM模型文件未找到"
        log_error "请先将BFM模型文件放到 ./bfm_models/ 目录"
        exit 1
    fi

    docker compose -f docker-compose.ernerf.yml run --rm ernerf convert-bfm

    if [ $? -eq 0 ]; then
        log_info "BFM模型转换成功"
    else
        log_error "BFM模型转换失败"
        exit 1
    fi
}

# 测试容器
test_container() {
    log_step "测试ER-NeRF容器..."

    docker compose -f docker-compose.ernerf.yml run --rm ernerf python -c "
import torch
import torch3d
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print('All tests passed!')
"

    if [ $? -eq 0 ]; then
        log_info "容器测试通过"
    else
        log_error "容器测试失败"
        exit 1
    fi
}

# 停止容器
stop_container() {
    log_step "停止ER-NeRF容器..."

    docker compose -f docker-compose.ernerf.yml down

    log_info "容器已停止"
}

# 查看状态
show_status() {
    log_step "ER-NeRF容器状态..."

    docker compose -f docker-compose.ernerf.yml ps
}

# 查看日志
show_logs() {
    docker compose -f docker-compose.ernerf.yml logs -f
}

# 完整设置
full_setup() {
    log_info "=================================="
    log_info "ER-NeRF Docker 完整设置"
    log_info "=================================="
    echo ""

    check_docker
    check_nvidia_docker
    check_bfm
    build_image
    start_container
    test_container

    echo ""
    log_info "=================================="
    log_info "设置完成！"
    log_info "=================================="
    echo ""
    log_info "现在可以："
    log_info "1. 启动Backend服务: export USE_DOCKER_FOR_ERNERF=true && python app.py"
    log_info "2. 或直接使用Docker: docker compose -f docker-compose.ernerf.yml run --rm ernerf preprocess <video.mp4>"
    echo ""
}

# 主函数
main() {
    case "${1:-help}" in
        check)
            check_docker
            check_nvidia_docker
            check_bfm
            ;;
        build)
            build_image
            ;;
        start)
            start_container
            ;;
        stop)
            stop_container
            ;;
        restart)
            stop_container
            start_container
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        setup-bfm)
            setup_bfm
            ;;
        test)
            test_container
            ;;
        setup)
            full_setup
            ;;
        shell)
            docker compose -f docker-compose.ernerf.yml run --rm ernerf shell
            ;;
        help|--help|-h)
            echo "ER-NeRF Docker 快速启动脚本"
            echo ""
            echo "用法: bash scripts/start_ernerf_docker.sh [命令]"
            echo ""
            echo "命令:"
            echo "  setup     - 完整设置（检查、构建、启动、测试）"
            echo "  check     - 检查环境（Docker、NVIDIA Docker、BFM模型）"
            echo "  build     - 构建Docker镜像"
            echo "  start     - 启动容器"
            echo "  stop      - 停止容器"
            echo "  restart   - 重启容器"
            echo "  status    - 查看容器状态"
            echo "  logs      - 查看容器日志"
            echo "  setup-bfm - 转换BFM模型"
            echo "  test      - 测试容器"
            echo "  shell     - 进入容器Shell"
            echo ""
            echo "示例:"
            echo "  bash scripts/start_ernerf_docker.sh setup"
            echo "  bash scripts/start_ernerf_docker.sh start"
            echo "  bash scripts/start_ernerf_docker.sh status"
            ;;
        *)
            log_error "未知命令: $1"
            echo "使用 'bash scripts/start_ernerf_docker.sh help' 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
