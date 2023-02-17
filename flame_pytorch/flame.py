"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

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
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues, lbs, vertices2landmarks
from smplx.utils import Struct, rot_mat_to_euler, to_np, to_tensor


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, "rb") as f:
            self.flame_model = Struct(**pickle.load(f, encoding="latin1"))
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros(
            [self.batch_size, 300 - config.shape_params],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter(
            "shape_betas", nn.Parameter(default_shape, requires_grad=False)
        )

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros(
            [self.batch_size, 100 - config.expression_params],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter(
            "expression_betas", nn.Parameter(default_exp, requires_grad=False)
        )

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros(
            [self.batch_size, 6], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        )

        default_neck_pose = torch.zeros(
            [self.batch_size, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        )

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

