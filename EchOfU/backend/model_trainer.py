import os
import subprocess
import shutil
import json
import numpy as np
from .path_manager import PathManager


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
            
            # 获取 ER-NeRF 源代码根目录
            er_nerf_root = pm.get_root_begin_path("ER-NeRF")
            
            # 预处理数据存放路径: models/ER-NeRF/data/<task_id>
            preprocess_data_path = pm.get_ernerf_data_path(task_id)
            pm.ensure_directory(preprocess_data_path)
            
           
            # 步骤 1: 数据预处理 (提取图像、音频、Landmarks)
           
            print(f"[backend.model_trainer] [1/3] 正在进行数据预处理: {ref_video_path}")
            
            process_script = os.path.join(er_nerf_root, "data_utils", "process.py")
            process_cmd = [
                "python", process_script,
                ref_video_path,
                "--task", task_id
                "--asr_model", "deepspeech" # 显式指定音频模型
            ]
            
            # 执行预处理
            subprocess.run(process_cmd, check=True)

            
            # 步骤 2: 动态计算当前进度
           
            print(f"[backend.model_trainer] [2/3] 计算当前训练进度...")
            
            # 1. 读取实际的数据集长度 (1个Epoch是多少步)
            json_path = os.path.join(preprocess_data_path, 'transforms_train.json')
            real_dataset_length = 100 # 默认值防止读取失败
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    transform_data = json.load(f)
                    real_dataset_length = len(transform_data['frames'])
            else:
                print("[backend.model_trainer] 警告：未找到 transforms_train.json，使用默认数据集长度计算")

            # 2. 扫描 Checkpoints 文件夹确定当前 Epoch
            ckpt_path = os.path.join(model_save_path, 'checkpoints')
            current_epoch = 0
            if os.path.exists(ckpt_path):
                files = [f for f in os.listdir(ckpt_path) if f.startswith('ngp_ep') and f.endswith('.pth')]
                if files:
                    # 文件名格式通常为 ngp_ep0010.pth，提取数字部分
                    try:
                        current_epoch = max([int(f[6:10]) for f in files])
                    except ValueError:
                        current_epoch = 0
            
            current_step = current_epoch * real_dataset_length
            print(f"[backend.model_trainer] 实际数据集长度: {real_dataset_length}, 当前Epoch: {current_epoch}, 当前总步数: {current_step}")
            
            
            # 步骤 3: 构造三阶段 train_cmd 并训练
          
            print(f"[backend.model_trainer] [3/3] 开始/继续 训练 ER-NeRF 模型...")

            # 定义每个阶段想要“跑多少步”
            stage1_duration = 70000  # 打底
            stage2_duration = 60000  # 嘴唇精调
            stage3_duration = 70000  # 躯干训练

            # 计算绝对的目标点 (累加)
            target_1 = stage1_duration
            target_2 = target_1 + stage2_duration
            target_3 = target_2 + stage3_duration
            
             # 基础参数
            base_train_args = [
                "python", os.path.join(er_nerf_root, "main.py"),
                preprocess_data_path,            # 数据路径
                "--workspace", model_save_path,  # 指定统一的模型保存路径 (Workspace)
                "-O",                            # 开启优化 (fp16, cuda_ray等)
                "--num_rays", "8192", 
                "--emb", 
                "--fp16", 
                "--exp_eye",
                "--asr_model", "deepspeech"      # 显式指定
            ]

            # GPU 设置
            env = os.environ.copy()
            if 'gpu_choice' in data:
                #  gpu_choice 格式为 "GPU0" -> "0"
                gpu_id = data['gpu_choice'].replace("GPU", "")
                env['CUDA_VISIBLE_DEVICES'] = gpu_id

            # 自动阶段判定逻辑
            train_cmd = None
            phase_msg = ""

            if current_step < target_1:
                # 阶段一：打底 (目标 70000)
                phase_msg = f"[阶段1] 正在进行基础训练，目标步数: {target_1}"
                train_cmd = base_train_args + ["--iters", str(target_1), "--lr", "1e-2"]

            elif current_step < target_2:
                # 阶段二：嘴唇 (目标 130000)
                # 注意：此时加载之前的存档，专门精调嘴部
                phase_msg = f"[阶段2] 正在精调嘴唇，目标步数: {target_2}"
                train_cmd = base_train_args + ["--iters", str(target_2), "--lr", "1e-4", "--finetune_lips"]

            elif current_step < target_3:
                # 阶段三：躯干 (目标 200000)
                # 此时开启 --torso，练出身体
                phase_msg = f"[阶段3] 正在训练躯干，目标步数: {target_3}"
                train_cmd = base_train_args + ["--iters", str(target_3), "--lr", "1e-4", "--torso"]
            else:
                phase_msg = "所有阶段已完成训练。"
                train_cmd = None

            print(phase_msg)

            if train_cmd:
            # 处理自定义参数 (如果前端传了，可能会覆盖上面的设置，需注意)
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
            
            # cwd 设置为 er_nerf_root，因为 main.py 内部有很多相对路径引用
            result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env,
                cwd=er_nerf_root 
            )
            
            # 打印最后几行日志以便调试
            print(f"[backend.model_trainer] 训练输出摘要:\n{result.stdout[-500:]}")
            print(f"[backend.model_trainer] ER-NeRF 训练阶段完成！")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] ER-NeRF 训练失败: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] 未知错误: {e}")
            return ref_video_path

    print("[backend.model_trainer] 训练流程结束")
    return ref_video_path




