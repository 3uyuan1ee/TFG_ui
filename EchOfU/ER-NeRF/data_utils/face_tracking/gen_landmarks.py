import os
import cv2
import numpy as np
import face_alignment

img_dir = 'data/obama/ori_imgs'
save_dir = 'data/obama/parsing'
os.makedirs(save_dir, exist_ok=True)

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device='cuda'
)

for img_name in sorted(os.listdir(img_dir)):
    if not img_name.endswith('.jpg'):
        continue
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    preds = fa.get_landmarks(img)
    if preds is None:
        continue
    lms = preds[0]
    np.save(
        os.path.join(save_dir, img_name.replace('.jpg', '.npy')),
        lms
    )

print('✅ landmarks generation finished')
