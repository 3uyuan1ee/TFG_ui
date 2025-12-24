

import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import face_alignment

# --- [参数配置] ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True)
parser.add_argument('--gt_path', type=str, required=True)
args = parser.parse_args()

TARGET_DIR = args.pred_path
GT_DIR = args.gt_path
SKIP_FRAMES = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_ssim_regional():
    ssim_global_list = []
    ssim_head_list = []

    # 1. 初始化人脸检测器 (用于自动定位头部范围)
    print(f"[INFO] 正在加载人脸检测器...")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False)

    # 获取文件
    all_files = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith('_rgb.png')])
    files = all_files[::SKIP_FRAMES]

    print(f"[INFO] 采样处理帧数: {len(files)}")

    for f_name in tqdm(files):
        try:
            parts = f_name.split('_')
            idx = int(parts[-2])

            pred = cv2.imread(os.path.join(TARGET_DIR, f_name))
            gt = None
            for fmt in [f"{idx}.jpg", f"{idx:04d}.jpg", f"{idx}.png", f"{idx:04d}.png"]:
                p_path = os.path.join(GT_DIR, fmt)
                if os.path.exists(p_path):
                    gt = cv2.imread(p_path)
                    break

            if pred is not None and gt is not None:
                # 尺寸对齐
                if pred.shape != gt.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

                # --- A. 计算全局 SSIM ---
                val_global = ssim(pred, gt, channel_axis=2)
                ssim_global_list.append(val_global)

                # --- B. 计算头部区域 SSIM ---
                # 在预测图中检测关键点
                preds_pts = fa.get_landmarks(pred)
                if preds_pts:
                    pts = preds_pts[0]
                    # 获取人脸包围盒 (x_min, y_min, x_max, y_max)
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)

                    # 向上扩展 50% 以包含额头和头发，向左右扩展 10%
                    h = y_max - y_min
                    w = x_max - x_min
                    y_min_head = max(0, int(y_min - 0.5 * h))
                    y_max_head = min(pred.shape[0], int(y_max + 0.1 * h))
                    x_min_head = max(0, int(x_min - 0.1 * w))
                    x_max_head = min(pred.shape[1], int(x_max + 0.1 * w))

                    # 裁剪出头部
                    crop_pred = pred[y_min_head:y_max_head, x_min_head:x_max_head]
                    crop_gt = gt[y_min_head:y_max_head, x_min_head:x_max_head]

                    if crop_pred.size > 0:
                        val_head = ssim(crop_pred, crop_gt, channel_axis=2)
                        ssim_head_list.append(val_head)
        except Exception as e:
            continue

    # 输出结果
    print("\n" + "=" * 45)
    if ssim_global_list:
        print(f"✅ 计算完成！有效样本: {len(ssim_global_list)}")
        print(f"📊 全局平均 SSIM (Full Body): {np.mean(ssim_global_list):.4f}")
        if ssim_head_list:
            print(f"📊 头部平均 SSIM (Head Only): {np.mean(ssim_head_list):.4f}")
            print(f"\n💡 分析：头部指标通常高于全局指标，说明核心面部重建质量更好。")
    else:
        print("❌ 未能成功计算，请检查路径。")
    print("=" * 45)

if __name__ == "__main__":
    run_ssim_regional()