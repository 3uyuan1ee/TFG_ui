import face_alignment
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch

# --- [核对路径] ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True, help='生成图片的路径')
parser.add_argument('--gt_path', type=str, required=True, help='真实图片的路径')
args = parser.parse_args()

TARGET_DIR = args.pred_path
GT_DIR = args.gt_path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 初始化检测器
print(f"[INFO] 正在加载模型至 {DEVICE}...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False)

# 2. 获取生成文件并测试一个路径
all_files = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith('_rgb.png')])
if not all_files:
    print(f"❌ 错误：在 {TARGET_DIR} 没找到任何 _rgb.png 文件")
    exit()

print(f"[INFO] 待处理文件总数: {len(all_files)}")
print(f"[DEBUG] 第一个文件名示例: {all_files[0]}")

lmd_total = []
m_lmd_total = []

# 3. 开始循环
for f_name in tqdm(all_files[::5]):  # 每5帧采一次
    try:
        # --- 核心逻辑：精准提取 0001 ---
        # 针对 ngp_ep0001_0001_rgb.png -> 提取出 0001
        idx_str = f_name.replace('_rgb.png', '').split('_')[-1]
        idx = int(idx_str)

        # --- 暴力尝试真值图路径 ---
        gt_file = None
        # 尝试：0001.jpg, 0001.png, 1.jpg, 1.png 等所有可能
        possible_names = [f"{idx_str}.jpg", f"{idx_str}.png", f"{idx}.jpg", f"{idx}.png"]
        for name in possible_names:
            p = os.path.join(GT_DIR, name)
            if os.path.exists(p):
                gt_file = p
                break

        if gt_file is None:
            # 如果找不到，打印出它尝试过的路径（只打印前几次，免得刷屏）
            if len(lmd_total) < 1:
                print(f"\n[DEBUG] 匹配失败！我尝试找了这些但都没找到: {possible_names} 在目录 {GT_DIR}")
            continue

        # 读取图片
        p_img = cv2.imread(os.path.join(TARGET_DIR, f_name))
        g_img = cv2.imread(gt_file)

        if p_img is None or g_img is None: continue

        p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)

        # 尺寸对齐
        if p_img.shape != g_img.shape:
            g_img = cv2.resize(g_img, (p_img.shape[1], p_img.shape[0]))

        # 检测关键点
        preds = fa.get_landmarks(p_img)
        gts = fa.get_landmarks(g_img)

        if preds and gts:
            p_pts = np.array(preds[0])
            g_pts = np.array(gts[0])

            # --- 关键：中心化补丁（消除 22.x 这种位移误差） ---
            # 扣除人脸中心的偏移，只算相对动作的误差
            p_pts_norm = p_pts - np.mean(p_pts, axis=0)
            g_pts_norm = g_pts - np.mean(g_pts, axis=0)

            lmd_total.append(np.mean(np.linalg.norm(p_pts_norm - g_pts_norm, axis=1)))
            m_lmd_total.append(np.mean(np.linalg.norm(p_pts_norm[48:68] - g_pts_norm[48:68], axis=1)))

    except Exception as e:
        continue

# 4. 输出
print("\n" + "=" * 40)
if lmd_total:
    print(f"✅ 匹配成功！有效样本: {len(lmd_total)}")
    print(f"📊 平均 LMD: {np.mean(lmd_total):.4f}")
    print(f"👄 平均 M-LMD: {np.mean(m_lmd_total):.4f}")
else:
    print("❌ 依然未能匹配。请手动确认以下信息：")
    print(f"1. 你的 GT 文件夹路径：{os.path.abspath(GT_DIR)}")
    print(f"2. 里面是不是有文件叫 '0001.jpg' 这种格式？")
print("=" * 40)