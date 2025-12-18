import os
import subprocess
import shutil
from path_manager import PathManager

def train_model(data):
    """
    模拟模型训练逻辑。
    负责调度 SyncTalk 或 ER-NeRF 的训练脚本。
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
            print("[backend.model_trainer] 开始 ER-NeRF 训练流程...")
            
            # 获取 ER-NeRF 源代码根目录
            er_nerf_root = pm.get_root_begin_path("ER-NeRF")
            
            # 预处理数据存放路径: models/ER-NeRF/data/<task_id>
            preprocess_data_path = pm.get_ernerf_data_path(task_id)
            pm.ensure_directory(preprocess_data_path)
            
           
            # 步骤 1: 数据预处理
            print(f"[backend.model_trainer] [1/2] 正在进行数据预处理: {ref_video_path}")
            
            process_script = os.path.join(er_nerf_root, "data_utils", "process.py")
            process_cmd = [
                "python", process_script,
                ref_video_path,
                "--task", task_id
            ]
            
            
            subprocess.run(process_cmd, check=True)
            
           
            # 步骤 2: 模型训练
            print(f"[backend.model_trainer] [2/2] 开始训练 ER-NeRF 模型...")
            
            # 获取训练轮数，默认为10 (前端文档标准)
            epochs = str(data.get('epoch', 10))
            
            train_script = os.path.join(er_nerf_root, "main.py")
            
            train_cmd = [
                "python", train_script,
                preprocess_data_path,            # 数据路径
                "--workspace", model_save_path,  # 指定统一的模型保存路径 (Workspace)
                "-O",                            # 优化参数
                "--iters", epochs,
                "--save_latest"
            ]

            # 处理自定义参数 (custom_params)
            # 前端格式示例: "lr=0.001" -> 解析为 "--lr 0.001"
            custom_params = data.get('custom_params', '')
            if custom_params:
                print(f"[backend.model_trainer] 解析自定义参数: {custom_params}")
                params_list = custom_params.split(',') # 假设逗号分隔
                for param in params_list:
                    if '=' in param:
                        key, value = param.split('=')
                        train_cmd.append(f"--{key.strip()}")
                        train_cmd.append(value.strip())

            # GPU 设置
            env = os.environ.copy()
            if 'gpu_choice' in data:
                # 假设 gpu_choice 格式为 "GPU0" -> "0"
                gpu_id = data['gpu_choice'].replace("GPU", "")
                env['CUDA_VISIBLE_DEVICES'] = gpu_id

            print(f"[backend.model_trainer] 执行训练命令: {' '.join(train_cmd)}")
            
            result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            print(f"[backend.model_trainer] ER-NeRF 训练成功！模型保存在: {model_save_path}")
            
            # 训练接口返回的 video_path 通常用于前端展示，如果训练本身不生成视频，
            # 可以返回参考视频路径，或者如果在 model_save_path 下生成了验证视频，则返回验证视频。
            
        
        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] ER-NeRF 训练失败: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] 未知错误: {e}")
            return ref_video_path

    print("[backend.model_trainer] 训练流程结束")
    return ref_video_path
