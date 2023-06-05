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

from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import os 
try:
    import cPickle as pickle
except ImportError:
    import pickle
from .utils import rodrigues , quat_feat , with_zeros , torch_fast_rotutils
from ..config import cfg 

class SUPR(nn.Module):
    def __init__(self,path_model,num_betas=10 , constrained = True):
        super(SUPR, self).__init__()

        if not os.path.exists(path_model):
            raise RuntimeError('Path does not exist %s' % (path_model))
        import numpy as np

        model = np.load(path_model,allow_pickle=True,encoding='latin1')[()]
        
        #Number of model joints 
        self.num_joints = model['kintree_table'].shape[1]

        #Number of model vertices 
        self.num_verts = model['v_template'].shape[0]


        #Loading the contrained Kinematic Tree, dictionary contains: 
        # Type of joints in the kinematic tree and the Constrained Axis of Rotation.
        if 'axis_meta' in model.keys():
            self.axis_meta = model['axis_meta']
            self.constrained = True  
            #Number of pose parameters for the cosntrained kinematic tree 
            self.num_pose = model['axis_meta']['num_pose']
        else:
            self.axis_meta = None 
            self.constrained = False 
            #Number of Joints x 3 
            self.num_pose = self.num_joints*3 
        
        J_regressor = model['J_regressor']
        self.num_betas = num_betas

        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor', torch.cuda.FloatTensor(J_regressor))

        # Model skinning weights
        self.register_buffer('weights', torch.cuda.FloatTensor(model['weights']))
        # Model pose corrective blend shapes
        self.register_buffer('posedirs', torch.cuda.FloatTensor(model['posedirs'].reshape((-1,self.num_joints*4))))
        # Mean Shape
        self.register_buffer('v_template', torch.cuda.FloatTensor(model['v_template']))
        # Shape corrective blend shapes
        self.register_buffer('shapedirs', torch.cuda.FloatTensor(np.array(model['shapedirs'][:,:,:num_betas])))
        # Mesh traingles
        self.register_buffer('faces', torch.from_numpy(model['f'].astype(np.int64)))
        self.f = model['f']
        # Kinematic tree of the model
        self.register_buffer('kintree_table', torch.from_numpy(model['kintree_table'].astype(np.int64)))

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.verts = None
        self.J = None
        self.R = None

    def display_info(self):
        '''
            Display Info about the model 
        '''

        if self.constrained:
            print('Kinematic Treee: Constrained')
        else:
            print('Kinematic Tree: Un-Constrained')

        print('Number of Pose Parameters: %d'%(self.num_pose))
        print('Number of Joints: %d'%(self.num_joints))
        print('Number of Vertices:%d'%(self.num_verts))

    def forward(self, pose, betas , trans):
        '''
            SUPR forward pass given pose, betas (shape) and trans
            return the model vertices and transformed joints
        :param pose: pose  parameters 
        :param beta: beta  parameters
        :param beta: trans parameters 
        '''

        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs  = self.shapedirs.view(-1, self.num_betas)[None, :].expand(batch_size, -1, -1)
        beta = betas[:, :, None]

        num_verts = v_template.shape[1]
        batch_size = pose.shape[0]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, num_verts, 3) + v_template
        num_joints = int(self.J_regressor.shape[0]/3)

        #Computing the shape correctives 
        pad_v_shaped = v_shaped.view(-1,num_verts*3)
        pad_v_shaped = torch.cat([pad_v_shaped,torch.ones((batch_size,1)).to(device)],axis=1)
        J = torch.einsum('ji,ai->aj', self.J_regressor, pad_v_shaped)
        J = J.view(-1,num_joints,3)
        
        
        # Replacing that with the Fast Rot Utils Module....
        if self.constrained and self.axis_meta is not None:
            torch_feat , R    = torch_fast_rotutils(pose,self.axis_meta)
            torch_feat = torch_feat.view((batch_size,-1))
        else:
            torch_feat = quat_feat(pose.view(-1,self.num_joints, 3)).view(batch_size, -1)
            R = rodrigues(pose.view(-1,self.num_joints, 3)).view(batch_size, num_joints, 3, 3)
            R = R.view(batch_size, num_joints, 3, 3)


        pose_feat = torch_feat
        posedirs = self.posedirs[None, :].expand(batch_size, -1, -1)

        #Computing the Pose-Depedent Corrective BlendShapes 
        v_posed = v_shaped + torch.matmul(posedirs, pose_feat[:, :, None]).view(-1, num_verts, 3)

        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, num_joints, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, num_joints):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, num_joints, 1).to(device)], dim=2).view(batch_size, num_joints, 4, 1)
        zeros = torch.zeros(batch_size, num_joints, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(num_joints, -1)).view(num_verts, batch_size, 4,4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        v = v + trans[:,None,:]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped

        root_transform = with_zeros(torch.cat((R[:,0],J[:,0][:,:,None]),2))
        results =  [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = with_zeros(torch.cat((R[:, i + 1], J[:, i + 1][:,:,None] - J[:, self.parent[i]][:,:,None]), 2))
            curr_res = torch.matmul(results[self.parent[i]],transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        v.J_transformed = posed_joints + trans[:,None,:]

        return v
