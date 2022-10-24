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
# Code Developed by:
# Ahmed A. A. Osman

from supr.tf.supr import SUPR
import tensorflow as tf
import numpy as np
batch_size = 10
gender = 'male'
 
path_model = '' 
supr = SUPR(path_model)
trans = tf.constant(np.zeros((1,3)),dtype=tf.float32)
pose = tf.constant(np.zeros((1,75*3)),dtype=tf.float32)
betas = tf.constant(np.zeros((1,10)),dtype=tf.float32)
vv =supr(pose,betas,trans)
import pdb;pdb.set_trace()

