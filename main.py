"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config

config = get_config()
radian = np.pi / 180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.zeros(8, 100).cuda()

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
pose_params_numpy = np.array(
    [
        [0.0, 30.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -30.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 85.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -48.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 10.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -15.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.0 * radian, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

# Cerating a batch of neutral expressions
expression_params = torch.zeros(8, 50, dtype=torch.float32).cuda()
flamelayer.cuda()

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
vertice, landmark = flamelayer(
    shape_params, expression_params, pose_params
)  # For RingNet project
print(vertice.size(), landmark.size())

if config.optimize_eyeballpose and config.optimize_neckpose:
