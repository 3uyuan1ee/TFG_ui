import os
import numpy as np
from scipy.io import loadmat

BFM_PATH = './3DMM/01_MorphableModel.mat'
SAVE_DIR = './3DMM'

bfm = loadmat(BFM_PATH)

# ========== Shape ==========
shapeMU = bfm['shapeMU'].reshape(-1)
shapePC = bfm['shapePC']
shapeEV = bfm['shapeEV'].reshape(-1)

np.save(
    os.path.join(SAVE_DIR, 'shape_para.npy'),
    {'mu': shapeMU, 'pc': shapePC, 'ev': shapeEV}
)

# ========== Texture ==========
texMU = bfm['texMU'].reshape(-1)
texPC = bfm['texPC']
texEV = bfm['texEV'].reshape(-1)

np.save(
    os.path.join(SAVE_DIR, 'tex_para.npy'),
    {'mu': texMU, 'pc': texPC, 'ev': texEV}
)

# ========== Expression (placeholder) ==========
# BFM 2009 has no expression PCA; ER-NeRF expects this file.
exp_dim = 50  # common convention
np.save(
    os.path.join(SAVE_DIR, 'exp_para.npy'),
    {
        'mu': np.zeros(exp_dim),
        'pc': np.zeros((shapePC.shape[0], exp_dim)),
        'ev': np.ones(exp_dim)
    }
)

print('✅ convert_BFM finished successfully.')
