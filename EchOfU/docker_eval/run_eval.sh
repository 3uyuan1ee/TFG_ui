#!/bin/bash

# 接收 Docker 传入的两个路径
GT_PATH="/data/gt"
PRED_PATH="/data/pred"

echo "=================================================="
echo "🚀 开始自动化评估 (Docker Mode)"
echo "Ground Truth 路径: $GT_PATH"
echo "Prediction   路径: $PRED_PATH"
echo "=================================================="

# --- 0. 准备工作：整理 rgb_only 文件夹 (为了 FID) ---
echo "[0/3] 正在整理图片数据..."
TEMP_RGB_DIR="/tmp/rgb_only"
mkdir -p $TEMP_RGB_DIR

# 这里的逻辑是：不管输入目录多乱，我把里面的 *_rgb.png 都拷出来
# 如果找不到 *_rgb.png，就拷贝所有 .png (容错)
count=$(find "$PRED_PATH" -name "*_rgb.png" | wc -l)
if [ "$count" -gt 0 ]; then
    echo "发现 $count 张 _rgb.png 图片，正在复制..."
    cp "$PRED_PATH"/*_rgb.png $TEMP_RGB_DIR/
else
    echo "⚠️ 没找到 _rgb.png，尝试复制所有 .png..."
    cp "$PRED_PATH"/*.png $TEMP_RGB_DIR/
fi

# --- 1. 运行 LMD 评估 ---
echo -e "\n[1/3] 正在运行 Landmark Distance (LMD) 评估..."
# 注意：这里我们传入原始路径，因为 LMD 代码里自己有 split('_rgb.png') 的逻辑
python LMD_eval.py --pred_path "$PRED_PATH" --gt_path "$GT_PATH"

# --- 2. 运行 SSIM 评估 ---
echo -e "\n[2/3] 正在运行 SSIM 评估..."
# SSIM 代码也用清洗后的文件夹比较稳妥，或者用原始的也行，这里用清洗后的
python SSIM_eval.py --pred_path "$TEMP_RGB_DIR" --gt_path "$GT_PATH"

# --- 3. 运行 FID 评估 ---
echo -e "\n[3/3] 正在运行 FID 评估..."
echo "计算生成分布: $TEMP_RGB_DIR"
echo "计算真实分布: $GT_PATH"
# 调用 pytorch-fid 库
python -m pytorch_fid "$GT_PATH" "$TEMP_RGB_DIR" --device cuda:0 --batch-size 64

echo "=================================================="
echo "✅ 所有评估任务已完成！"