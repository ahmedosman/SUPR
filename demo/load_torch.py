# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
#
# Code Developed by:
# Ahmed A. A. Osman
from supr.pytorch.supr import SUPR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
num_betas=10
batch_size=1

#Set the Path of the model.
path_model = ''
if path_model is '':
    raise RuntimeError('Set the model path!!')

#Loading the Model 
m = SUPR(path_model,num_betas=num_betas)
#Printing Information about the loaded model - 
# 1 - dimensionality of the pose/shape parameters
# 2- Kinematic Tree (Constrained or Un Constrained )
m.display_info()

#Passing Pose and Shape Parameters 
poses = torch.cuda.FloatTensor(np.zeros((batch_size,m.num_pose)))
betas = torch.cuda.FloatTensor(betas)
trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
model = m.forward(poses, betas,trans)