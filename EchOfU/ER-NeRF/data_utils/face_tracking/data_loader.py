
import os
import torch
import numpy as np


def load_dir(path, start, end):
    lmss = []
    imgs_paths = []

    # --- 路径兼容逻辑开始 ---
    # 无论传入的 path 是 'data/obama' 还是 'data/obama/ori_imgs'
    # 我们都统一获取它的父级目录作为 base_dir
    path = path.rstrip('/')
    if path.endswith('ori_imgs'):
        base_dir = os.path.dirname(path)
    else:
        base_dir = path

    # 强制指定 landmarks 和 ori_imgs 的绝对位置
    lms_dir = os.path.join(base_dir, 'landmarks')
    img_dir = os.path.join(base_dir, 'ori_imgs')
    # --- 路径兼容逻辑结束 ---

    print(f"[INFO] Loading landmarks from: {lms_dir}")
    print(f"[INFO] Loading images from: {img_dir}")

    for i in range(start, end):
        # 尝试匹配两种常见的文件名格式：数字.npy 或 数字.jpg.npy
        lms_path = os.path.join(lms_dir, str(i) + ".npy")
        if not os.path.isfile(lms_path):
            lms_path = os.path.join(lms_dir, str(i) + ".jpg.npy")

        img_path = os.path.join(img_dir, str(i) + ".jpg")

        if os.path.isfile(lms_path):
            # 使用 np.load 加载二进制文件
            try:
                lms = np.load(lms_path).astype(np.float32)
                # 如果 npy 维度是 (1, 68, 2) 或 (68, 2)，统一处理
                lms = lms.squeeze()
                lmss.append(lms)
                imgs_paths.append(img_path)
            except Exception as e:
                print(f"[WARN] Failed to load {lms_path}: {e}")
                continue

    if len(lmss) == 0:
        print(f"❌ 错误：在以下路径未找到有效数据：")
        print(f"   特征点目录: {os.path.abspath(lms_dir)}")
        print(f"   图片目录: {os.path.abspath(img_dir)}")
        print(f"请检查文件名是否为 '{start}.npy' 这种格式。")
        raise ValueError('need at least one array to stack')

    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()

    print(f"[INFO] Successfully loaded {len(lmss)} frames.")
    return lmss, imgs_paths