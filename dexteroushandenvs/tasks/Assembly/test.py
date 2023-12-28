
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

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

def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


device = "cuda:0"
args = gymutil.parse_arguments(description="test",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])


gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 10
sim_params.substeps = 2
physics_engine = gymapi.SIM_PHYSX
if physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

sim_params.use_gpu_pipeline = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
asset_root = "/mnt/data/pyCharm-project/Reinforcement_Learning/RL_assembly/assets"

# joint_asset_options = gymapi.AssetOptions()
# joint_asset_options.flip_visual_attachments = False
# joint_asset_options.fix_base_link = True
# joint_asset_options.disable_gravity = True
# joint_asset_options.thickness = 0.001
# joint_asset_options.angular_damping = 0.01
# joint_asset_file = "urdf/rokae_description/multi_joint.urdf"
# joint_asset = gym.load_asset(sim, asset_root, joint_asset_file, joint_asset_options)

# joint_pose = gymapi.Transform()
# joint_pose.p = gymapi.Vec3(0, 0.6, 0.1)
# joint_pose.r = gymapi.Quat(0, 0, 0, 1)

# joint_dof_props = gym.get_asset_dof_properties(joint_asset)
# joint_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
# joint_dof_props["stiffness"][:].fill(0.0)
# joint_dof_props["damping"][:].fill(0.0)

asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = False
asset_options.disable_gravity = True
asset_options.thickness = 0.001
asset_options.angular_damping = 0.01

if args.physics_engine == gymapi.SIM_PHYSX:
    asset_options.use_physx_armature = True

robot_asset_files_dict = {
            "ur5e": "urdf/ur5e_description/export_urdf/ur5e.urdf", #ur5e_description/robots/ur5e.urdf
            "rg2ft":"urdf/ur5e_description/export_urdf/rg2ft.urdf",
            "franka": "urdf/franka_description/robots/franka_panda_allegro.urdf",
            "onrobotrg2":"urdf/ur5e_description/export_urdf/rg2.urdf",
            "large":  "urdf/xarm6/xarm6_allegro_left_fsr_large.urdf",
            "rokae": "urdf/rokae_description/xMatePro7.urdf",
            "franka_panda": "urdf/franka_panda/panda.urdf"
        }
arm_hand_asset_file = robot_asset_files_dict["rokae"]
arm_hand_asset = gym.load_asset(sim, asset_root, arm_hand_asset_file, asset_options)

rokea_dof_props = gym.get_asset_dof_properties(arm_hand_asset)
rokea_lower_limits = rokea_dof_props['lower']
rokea_upper_limits = rokea_dof_props['upper']
rokea_ranges = rokea_upper_limits - rokea_lower_limits
rokea_mids = 0.5 * (rokea_upper_limits + rokea_lower_limits)
rokea_num_dofs = len(rokea_dof_props)

# set default DOF states
default_dof_state = np.zeros(rokea_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:7] = np.array([0.0, 0.52, 0.0, 1.04, 0.0, 1.57, 0.261])

# set DOF control properties (except grippers)
# rokea_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
# rokea_dof_props["stiffness"][:7].fill(0.0)
# rokea_dof_props["damping"][:7].fill(0.0)

# # set DOF control properties for grippers
# rokea_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
# rokea_dof_props["stiffness"][7:].fill(800.0)
# rokea_dof_props["damping"][7:].fill(40.0)

num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default franka pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []
hand_indices = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add franka
    rokea_handle = gym.create_actor(env, arm_hand_asset, pose, "rokea", i, 1)

    # Set initial DOF states
    gym.set_actor_dof_states(env, rokea_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, rokea_handle, rokea_dof_props)

    # Get inital hand pose
    # hand_handle = gym.find_actor_rigid_body_handle(env, rokea_handle, "dh_hand")
    # hand_pose = gym.get_rigid_transform(env, hand_handle)
    # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    # init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # # Get global index of hand in rigid body state tensor
    # hand_idx = gym.find_actor_rigid_body_index(env, rokea_handle, "panda_hand", gymapi.DOMAIN_SIM)
    # hand_idxs.append(hand_idx)

   
    # mjoint_handle = gym.create_actor(env, joint_asset, joint_pose, "mjoint", i, 1)
    # gym.set_actor_dof_properties(env, mjoint_handle, joint_dof_props)
    # gym.enable_actor_dof_force_sensors(env, mjoint_handle)
    # z_indx = gym.find_actor_rigid_body_index(env, mjoint_handle, "z_link", gymapi.DOMAIN_ENV)
    # z_handle = gym.find_actor_rigid_body_handle(env, mjoint_handle, "z_link")

    # arm_idx = gym.get_actor_index(env, mjoint_handle, gymapi.DOMAIN_SIM)
    # hand_indices.append(arm_idx)
grid_count = gym.get_env_rigid_body_count(envs[0])

# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

hand_indices = to_torch(hand_indices, dtype=torch.int32, device=device)

force = gymapi.Vec3(0, 0, 1000.0)
torque_amt = 100
itr = 0
while not gym.query_viewer_has_closed(viewer):

    # Randomize desired hand orientations
    # if itr % 250 == 0 and args.orn_control:
    #     orn_des = torch.rand_like(orn_des)
    #     orn_des /= torch.norm(orn_des)

    itr += 1

    # forces = torch.zeros((num_envs, grid_count, 3), device=device, dtype=torch.float)
    # torques = torch.zeros((num_envs, grid_count, 3), device=device, dtype=torch.float)
    # forces[:, z_indx, 2] = -10
    # torques[:, 0, 2] = torque_amt
    # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)



    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_dof_force_tensor(sim)

    # targets = torch.ones((num_envs, 1), dtype=torch.float, device=device) * 0.05
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(targets))

    # joint_effort = gym.acquire_dof_force_tensor(sim)
    # joint_eff_data = gymtorch.wrap_tensor(joint_effort).view(num_envs, -1)
    # # print(joint_eff_data)

    # state_tensor = gym.acquire_rigid_body_state_tensor(sim)
    # state_data = gymtorch.wrap_tensor(state_tensor).view(num_envs, -1, 13)
    # z_link_state = state_data[:, z_indx, :]
    # linear_v = z_link_state[:, 10:]
    # print(linear_v)


    # Get current hand poses

    # gym.apply_body_forces(env, z_handle, force)
    # gym.apply_rigid_body_force_tensors(sim, z_handle, force, torque=None,space=gymapi.GLOBAL_SPACE)
    
    
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    # gym.sync_frame_time(sim)

# init_pose = to_torch(np.random.random([32, 7]))
# target_pose = to_torch(np.random.random([32, 7]))
# target_pose2 = to_torch(np.random.random([32, 7]))
# target_pose = torch.zeros([32, 7]).to(device)
# target_pose[:, :3] = init_pose[:, :3].clone()
# target_pose[:, 0] = to_torch(nut_rand_p[:, 0]).to(device)
# target_pose[:, 1] = to_torch(nut_rand_p[:, 1]).to(device)
# target_pose[:, 2] = to_torch(nut_rand_p[:, 2] + 0.2).to(device)
# target_pose[:, 3:7]  = to_torch([0, 0, 0, 1]).to(device) #init_pose[:, 3:7]

# path_generate([init_pose, target_pose, target_pose2])