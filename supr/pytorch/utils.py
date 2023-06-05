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
# Code Developed by:
# Ahmed A. A. Osman

import torch


def with_zeros(input):
    '''
      Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor

    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    '''
    batch_size  = input.shape[0]
    row_append     = torch.cuda.FloatTensor(([0.0, 0.0, 0.0, 1.0]))
    row_append.requires_grad = False
    padded_tensor     = torch.cat([input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor

def quat2mat(quat):
    '''
      Convert a quaternion to rotation matrices
    '''
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=2, keepdim=True)
    w, x, y, z = norm_quat[:,:, 0], norm_quat[:,:, 1], norm_quat[:,:, 2], norm_quat[:,:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=2).view(B,-1, 3, 3)
    return rotMat


def quat_feat(theta):
    '''
        Computes a normalized quaternion ([0,0,0,0]  when the body is in rest pose)
        given joint angles
    :param theta: A tensor of joints axis angles, batch size x number of joints x 3
    :return:
    '''
    l1norm = torch.norm(theta + 1e-8, p=2, dim=2)
    angle = torch.unsqueeze(l1norm, -1)    
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized,v_cos-1], dim=2)
    return quat

def rodrigues(theta):
    '''
        Computes the rodrigues representation given joint angles

    :param theta: batch_size x number of joints x 3
    :return: batch_size x number of joints x 3 x 4
    '''
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 2)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 2)

    return quat2mat(quat)

def torch_compute_rot_hinge(pose,axis):
    '''
      Quaternion multiplication 

    '''
    cos_feat = torch.cos(pose)
    sin_feat = torch.sin(pose)
    ax = axis[:,:,0][:,:,None]
    ay = axis[:,:,1][:,:,None]
    az = axis[:,:,2][:,:,None]

    row1 = torch.cat([ax**2.0+cos_feat*(1-ax**2.0), ax*ay*(1-cos_feat)+az*sin_feat  , ax*az*(1-cos_feat)-ay*sin_feat],dim=2)
    row2 = torch.cat([ax*ay*(1-cos_feat)-az*sin_feat, ay**2.0+cos_feat*(1-ay**2.0)  , ay*az*(1-cos_feat)+ax*sin_feat ],dim=2)
    row3 = torch.cat([ax*az*(1-cos_feat)+ay*sin_feat,ay*az*(1-cos_feat)-ax*sin_feat  ,  az**2.0+cos_feat*(1-az**2.0) ],dim=2)
    rot_mat = torch.stack([row1,row2,row3],dim=-1)
    rot_mat = rot_mat.permute([0,1,3,2])
    return rot_mat

def torch_quaternion_multiply(q1,q2):
    '''
       Torch Quaternion Multiply 
    '''
    x0 = q1[:,:,0]
    y0 = q1[:,:,1]
    z0 = q1[:,:,2]
    w0 = q1[:,:,3]
    x1 = q2[:,:,0]
    y1 = q2[:,:,1]
    z1 = q2[:,:,2]
    w1 = q2[:,:,3]
    xr = x1*w0 + y1*z0 - z1*y0 + w1*x0
    yr = -x1*z0 + y1*w0+z1*x0+w1*y0
    zr = x1*y0 - y1*x0 + z1*w0 + w1*z0
    wr = -x1*x0-y1*y0-z1*z0+w1*w0
    quat_mult = torch.stack([xr,yr,zr,wr],dim=2)
    return quat_mult



def torch_fast_rotutils(pose,meta):
    '''
        Compute the kinematic tree joint rotations matrices and the corresponding normalized 
        quaternion features 

        returns: rotation matrices, normalized quaternion features  
    '''
    import numpy as np 
    #Index of the pose parameters corresponding to spherical joints, 3 parameter per joints
    indx_spherical = meta['indx_spherical']
    #Index of the pose parameters corresponding to double hinge joints 
    indx_double_hinge1 = meta['indx_double_hinge1']
    #Index double hinge axis 
    indx_double_hinge1_axis = meta['indx_double_hinge1_axis']
    indx_double_hinge2 = meta['indx_double_hinge2']
    indx_double_hinge2_axis = meta['indx_double_hinge2_axis']
    indx_hinge = meta['indx_hinge']
    indx_hinge_axis = meta['indx_hinge_axis']
    reverse_indx = meta['reverse_indx']

    axis_indx = meta['axis_indx']
    axis = torch.cuda.FloatTensor(meta['axis'][:,axis_indx,:])
    indx_spherical = np.concatenate(indx_spherical)
    indx_hinge     = np.concatenate(indx_hinge,axis=0)
    
    spherical_pose = pose[:,indx_spherical]
    hinge_pose = pose[:,indx_hinge]
    hinge_axis_pose =  axis[:,indx_hinge_axis]
    
    hinge_pose1 = pose[:,indx_double_hinge1]
    hinge_axis_pose1 = axis[:,indx_double_hinge1_axis]
    
    hinge_pose2 = pose[:,indx_double_hinge2]
    hinge_axis_pose2 = axis[:,indx_double_hinge2_axis]

    list_rotat_mat = []
    list_feat = []
    
    num_joints = len(meta['indx_spherical'])
    list_rotat_mat.append(rodrigues(spherical_pose.reshape((-1,num_joints,3))))
    quaternion_axis = quat_feat(spherical_pose.reshape((-1,num_joints,3)))
    list_feat.append(quaternion_axis)


    rot_mat = torch_compute_rot_hinge(hinge_pose[:,:,None],hinge_axis_pose)
    test_case = [rot_mat,hinge_pose,hinge_axis_pose]
    list_rotat_mat.append(rot_mat)
    axis_angle = hinge_axis_pose*hinge_pose[:,:,None]
    
    num_joints = len(indx_hinge)
    quaternion_axis = quat_feat(axis_angle)
    list_feat.append(quaternion_axis)

    joint_pose1 =  hinge_pose1[:,:,None]
    joint_pose2 = hinge_pose2[:,:,None]

    axis_pose1  = hinge_axis_pose1
    axis_pose2  = hinge_axis_pose2

    axis_angle1 = axis_pose1*joint_pose1



    rot_mat1    = torch_compute_rot_hinge(joint_pose1, axis_pose1)
    num_joints1 = len(indx_double_hinge1)
    quat1       = quat_feat(axis_angle1)


    axis_angle2 = axis_pose2*joint_pose2
    rot_mat2    = torch_compute_rot_hinge(joint_pose2, axis_pose2)
    num_joints2 = len(indx_double_hinge2)
    quat2       = quat_feat(axis_angle2)
    
    quat_multiply = torch_quaternion_multiply(torch.reshape(quat1,[-1,num_joints1,4]), torch.reshape(quat2,[-1,num_joints2,4]))
    rot_mat       = torch.matmul(rot_mat1, rot_mat2)

    list_feat.append(quat_multiply)
    list_rotat_mat.append(rot_mat)

    torch_rot_mat = torch.cat(list_rotat_mat,dim=1)
    torch_feat    = torch.cat(list_feat,dim=1)

    torch_feat    = torch_feat[:,reverse_indx,:]
    torch_rot_mat = torch_rot_mat[:,reverse_indx,:,:]
    return   torch_feat , torch_rot_mat 

