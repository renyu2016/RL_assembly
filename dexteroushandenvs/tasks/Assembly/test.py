from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2

from einops import rearrange
import pickle
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import math

import numpy as np
import os
import torch




def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def path_generate(pose_seq):
        g_pth = []
        g_steps = []
        per_step = 0.01

        # start_point = pose_seq[0]
        # start_pos = start_point[:3]
        # start_orn = start_point[3:7]
    
        for i in range(1, len(pose_seq)):
            phase_start = np.array(pose_seq[i - 1].cpu())
            phase_end = np.array(pose_seq[i].cpu())


            s_pos = phase_start[:, :3]
            e_pos = phase_end[:, :3]

            s_orn = phase_start[:, 3:7]
            e_orn = phase_end[:, 3:7]

            interp_function = interp1d([0, 1], np.array([np.array(s_orn), np.array(e_orn)]), axis=0, kind='slinear')

            

            direct_vec = e_pos-s_pos
            norm_scale = np.linalg.norm(direct_vec, axis=1, keepdims=True)
            steps = np.array((norm_scale // per_step + 1))

            direct_vec /= norm_scale
            max_step = int(np.max(steps))
            actions_buf = np.zeros([phase_start.shape[0], int(max_step), phase_start.shape[1]])

            for n in range(int(max_step)):
                unachieve_ids = np.where(n < steps-1)[0]
                achieved_ids = np.where(n >= steps-1)[0]

                cur_act_buf = np.zeros([phase_start.shape[0], phase_start.shape[1]])
                cur_act_buf[unachieve_ids, :3] = s_pos[unachieve_ids, :] + n * per_step * direct_vec[unachieve_ids, :]
                cur_act_buf[achieved_ids, :3] = e_pos[achieved_ids, :]

                interp_query = n/steps[:, 0]
                interp_query[np.where(interp_query > 1)] = 1
                interp_orn = interp_function(interp_query)
                cur_act_buf[unachieve_ids, 3:7] = interp_orn[unachieve_ids, unachieve_ids, :]
                cur_act_buf[achieved_ids, 3:7] = e_orn[achieved_ids, :]
                actions_buf[:, n, :] = cur_act_buf
            g_pth.append(actions_buf)
            print(max_step)
        total_pth = np.concatenate(g_pth, axis=1)
        total_pth = to_torch(total_pth)
        return total_pth



device = "cuda:0"


init_pose = to_torch(np.random.random([32, 7]))
target_pose = to_torch(np.random.random([32, 7]))
target_pose2 = to_torch(np.random.random([32, 7]))
# target_pose = torch.zeros([32, 7]).to(device)
# target_pose[:, :3] = init_pose[:, :3].clone()
# target_pose[:, 0] = to_torch(nut_rand_p[:, 0]).to(device)
# target_pose[:, 1] = to_torch(nut_rand_p[:, 1]).to(device)
# target_pose[:, 2] = to_torch(nut_rand_p[:, 2] + 0.2).to(device)
# target_pose[:, 3:7]  = to_torch([0, 0, 0, 1]).to(device) #init_pose[:, 3:7]

path_generate([init_pose, target_pose, target_pose2])