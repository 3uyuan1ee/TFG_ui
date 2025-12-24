import os
import subprocess
import shutil
import json
import numpy as np
from .path_manager import PathManager
from .ernerf_docker_client import get_ernerf_docker_client

# ==============================================================================
# ER-NeRF 调用模式配置
# ==============================================================================
# 设置为 True 使用 Docker 调用 ER-NeRF
# 设置为 False 使用直接调用（需要本地环境）
USE_DOCKER_FOR_ERNERF = os.environ.get('USE_DOCKER_FOR_ERNERF', 'true').lower() == 'true'


def train_model(data):
    """
    模拟模型训练逻辑。
    负责调度 SyncTalk 或 ER-NeRF 的训练脚本。
    ER-NeRF 实现了三阶段自动训练逻辑。
    """
    # 初始化路径管理器
    pm = PathManager()
    
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    # 路径配置
    ref_video_path = data['ref_video']
    model_choice = data['model_choice']

    # 标准化模型名称（处理不同写法：ER_NeRF, ER NeRF, ER-NeRF）
    if model_choice in ['ER_NeRF', 'ER NeRF']:
        model_choice = 'ER-NeRF'
    elif model_choice == 'SyncTalk':
        model_choice = 'SyncTalk'
    
    # 获取任务ID (优先使用 speaker_id，否则使用文件名)
    if data.get('speaker_id'):
        task_id = data['speaker_id']
    else:
        # 如果是文件路径，提取文件名作为ID
        task_id = os.path.splitext(os.path.basename(ref_video_path))[0]

    # 1. 统一模型保存路径: TFG_ui/EchOfU/models/ER-NeRF/<task_id>
    # 使用 PathManager 获取 ER-NeRF 模型路径
    model_save_path = pm.get_ernerf_model_path(task_id)
    pm.ensure_directory(model_save_path)

    print(f"[backend.model_trainer] 任务ID: {task_id}, 目标模型路径: {model_save_path}")
    print("[backend.model_trainer] 模型训练中...")

    if model_choice == "SyncTalk":
        # SyncTalk 逻辑 (脚本通常在项目根目录的 SyncTalk 文件夹下)
        synctalk_script = pm.get_root_begin_path("SyncTalk", "run_synctalk.sh")
        
        try:
            cmd = [
                synctalk_script, "train",
                "--video_path", ref_video_path,
                "--gpu", str(data.get('gpu_choice', '0').replace("GPU", "")),
                "--epochs", str(data.get('epoch', 10))
            ]
            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("[backend.model_trainer] SyncTalk 训练输出:", result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] SyncTalk 训练失败: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] SyncTalk 错误: {e}")
            return ref_video_path

    elif model_choice == "ER-NeRF":
        try:
            print("[backend.model_trainer] 开始 ER-NeRF 预处理和训练流程...")
            print(f"[backend.model_trainer] 使用模式: {'Docker' if USE_DOCKER_FOR_ERNERF else '直接调用'}")

            # 预处理数据存放路径
            preprocess_data_path = pm.get_ernerf_data_path(task_id)
            pm.ensure_directory(preprocess_data_path)

            if USE_DOCKER_FOR_ERNERF:
                # Docker模式
                print("[backend.model_trainer] 使用Docker模式调用ER-NeRF...")
                try:
                    client = get_ernerf_docker_client()

                    # 步骤1: 数据预处理
                    print(f"[backend.model_trainer] [1/3] 正在进行数据预处理: {ref_video_path}")

                    # 获取相对路径（Docker容器内路径）
                    # 需要确保视频在project_root/data目录下可访问
                    if not os.path.isabs(ref_video_path):
                        ref_video_abs = os.path.abspath(ref_video_path)
                    else:
                        ref_video_abs = ref_video_path

                    success, message = client.preprocess(ref_video_abs, task_id)

                    if not success:
                        print(f"[backend.model_trainer] Docker预处理失败: {message}")
                        return ref_video_path

                    print(f"[backend.model_trainer] 预处理完成: {message}")

                    # 获取数据路径（Docker客户端格式）
                    docker_data_path = f"data/{task_id}"

                    # 步骤2: 计算训练进度
                    print(f"[backend.model_trainer] [2/3] 计算当前训练进度...")
                    json_path = os.path.join(preprocess_data_path, 'transforms_train.json')
                    real_dataset_length = 100
                    current_epoch = 0

                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            transform_data = json.load(f)
                            real_dataset_length = len(transform_data['frames'])

                    ckpt_path = os.path.join(model_save_path, 'checkpoints')
                    if os.path.exists(ckpt_path):
                        files = [f for f in os.listdir(ckpt_path) if f.startswith('ngp_ep') and f.endswith('.pth')]
                        if files:
                            try:
                                current_epoch = max([int(f[6:10]) for f in files])
                            except ValueError:
                                current_epoch = 0

                    current_step = current_epoch * real_dataset_length
                    print(f"[backend.model_trainer] 数据集长度: {real_dataset_length}, Epoch: {current_epoch}, 总步数: {current_step}")

                    # 步骤3: 训练
                    print(f"[backend.model_trainer] [3/3] 开始/继续训练ER-NeRF模型...")

                    # 获取GPU ID
                    gpu_id = 0
                    if 'gpu_choice' in data:
                        gpu_id = int(data['gpu_choice'].replace("GPU", ""))

                    # 使用auto模式（自动完成所有阶段）
                    success, message = client.train(
                        data_path=docker_data_path,
                        model_path=f"models/ER-NeRF/{task_id}",
                        gpu_id=gpu_id,
                        stage="auto"
                    )

                    if success:
                        print(f"[backend.model_trainer] ER-NeRF Docker训练完成: {message}")
                    else:
                        print(f"[backend.model_trainer] ER-NeRF Docker训练失败: {message}")
                        return ref_video_path

                except Exception as e:
                    print(f"[backend.model_trainer] ER-NeRF Docker训练异常: {e}")
                    import traceback
                    traceback.print_exc()
                    return ref_video_path

            else:
                # 直接调用模式（原有逻辑）
                print("[backend.model_trainer] 使用直接调用模式...")
                er_nerf_root = pm.get_root_begin_path("ER-NeRF")

                # 步骤 1: 数据预处理
                print(f"[backend.model_trainer] [1/3] 正在进行数据预处理: {ref_video_path}")

                process_script = os.path.join(er_nerf_root, "data_utils", "process.py")
                process_cmd = [
                    "python", process_script,
                    ref_video_path,
                    "--task", str(task_id),  # 确保task_id是字符串
                    "--asr_model", "deepspeech"
                ]

                subprocess.run(process_cmd, check=True)

                # 步骤 2: 动态计算当前进度
                print(f"[backend.model_trainer] [2/3] 计算当前训练进度...")

                json_path = os.path.join(preprocess_data_path, 'transforms_train.json')
                real_dataset_length = 100

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        transform_data = json.load(f)
                        real_dataset_length = len(transform_data['frames'])
                else:
                    print("[backend.model_trainer] 警告：未找到 transforms_train.json")

                ckpt_path = os.path.join(model_save_path, 'checkpoints')
                current_epoch = 0
                if os.path.exists(ckpt_path):
                    files = [f for f in os.listdir(ckpt_path) if f.startswith('ngp_ep') and f.endswith('.pth')]
                    if files:
                        try:
                            current_epoch = max([int(f[6:10]) for f in files])
                        except ValueError:
                            current_epoch = 0

                current_step = current_epoch * real_dataset_length
                print(f"[backend.model_trainer] 数据集长度: {real_dataset_length}, Epoch: {current_epoch}, 总步数: {current_step}")

                # 步骤 3: 构造三阶段训练命令
                print(f"[backend.model_trainer] [3/3] 开始/继续训练ER-NeRF模型...")

                stage1_duration = 70000
                stage2_duration = 60000
                stage3_duration = 70000

                target_1 = stage1_duration
                target_2 = target_1 + stage2_duration
                target_3 = target_2 + stage3_duration

                base_train_args = [
                    "python", os.path.join(er_nerf_root, "main.py"),
                    preprocess_data_path,
                    "--workspace", model_save_path,
                    "-O",
                    "--num_rays", "8192",
                    "--emb",
                    "--fp16",
                    "--exp_eye",
                    "--asr_model", "deepspeech"
                ]

                env = os.environ.copy()
                if 'gpu_choice' in data:
                    gpu_id = data['gpu_choice'].replace("GPU", "")
                    env['CUDA_VISIBLE_DEVICES'] = gpu_id

                # 自动阶段判定
                train_cmd = None
                phase_msg = ""

                if current_step < target_1:
                    phase_msg = f"[阶段1] 正在进行基础训练，目标步数: {target_1}"
                    train_cmd = base_train_args + ["--iters", str(target_1), "--lr", "1e-2"]
                elif current_step < target_2:
                    phase_msg = f"[阶段2] 正在精调嘴唇，目标步数: {target_2}"
                    train_cmd = base_train_args + ["--iters", str(target_2), "--lr", "1e-4", "--finetune_lips"]
                elif current_step < target_3:
                    phase_msg = f"[阶段3] 正在训练躯干，目标步数: {target_3}"
                    train_cmd = base_train_args + ["--iters", str(target_3), "--lr", "1e-4", "--torso"]
                else:
                    phase_msg = "所有阶段已完成训练。"

                print(phase_msg)

                if train_cmd:
                    custom_params = data.get('custom_params', '')
                    if custom_params:
                        print(f"[backend.model_trainer] 附加自定义参数: {custom_params}")
                        params_list = custom_params.split(',')
                        for param in params_list:
                            if '=' in param:
                                key, value = param.split('=')
                                train_cmd.append(f"--{key.strip()}")
                                train_cmd.append(value.strip())

                    print(f"[backend.model_trainer] 完整命令: {' '.join(train_cmd)}")

                    result = subprocess.run(
                        train_cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        env=env,
                        cwd=er_nerf_root
                    )

                    print(f"[backend.model_trainer] 训练输出摘要:\n{result.stdout[-500:]}")
                    print(f"[backend.model_trainer] ER-NeRF 训练阶段完成！")

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] ER-NeRF 训练失败: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] 未知错误: {e}")
            import traceback
            traceback.print_exc()
            return ref_video_path

    print("[backend.model_trainer] 训练流程结束")
    return ref_video_path




