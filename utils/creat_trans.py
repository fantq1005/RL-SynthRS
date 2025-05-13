import trimesh
import numpy as np
import random
import pyrender
from scipy.spatial.transform import Rotation as R

def set_object(x, y, z, factor, rotation_mode="random", fixed_angle=0):
    tx = x
    ty = y
    tz = z
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0], [0, 0, 0])
    translation_matrix = trimesh.transformations.translation_matrix([tx, ty, tz])
    scale_factor = factor
    scale_matrix = np.eye(4) * scale_factor
    scale_matrix[3, 3] = 1
    if rotation_mode == "random":
        rotation_angle_z = np.radians(random.uniform(0, 360))  # 随机角度
    elif rotation_mode == "fixed":
        rotation_angle_z = np.radians(fixed_angle)  # 固定角度
    else:
        raise ValueError("Invalid rotation mode. Use 'random' or 'fixed'.")
    rotation_matrix_z = trimesh.transformations.rotation_matrix(rotation_angle_z, [0, 0, 1], [0, 0, 0])
    combined_transform = np.dot(translation_matrix, np.dot(rotation_matrix_z, np.dot(scale_matrix, rotation_matrix_x)))
    return combined_transform

def compute_rotation_matrix(yaw, pitch):
    # 创建绕Y轴旋转的旋转矩阵（Yaw）
    rotation_matrix_yaw = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    # 创建绕X轴旋转的旋转矩阵（Pitch）
    rotation_matrix_pitch = trimesh.transformations.rotation_matrix(pitch, [1, 0, 0])

    # 将两个旋转矩阵相乘以组合旋转效果
    # 注意旋转的顺序，先绕Y轴旋转（Yaw），然后绕X轴旋转（Pitch）
    combined_rotation_matrix = np.dot(rotation_matrix_pitch, rotation_matrix_yaw)

    # 返回组合旋转矩阵的前3x3部分，因为trimesh生成的是4x4的仿射变换矩阵
    return combined_rotation_matrix[:3, :3]


def generate_rotated_pose_matrix(x, y, z):
    # 相机的初始位姿矩阵
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 原始方向向量和目标方向向量
    v_original = np.array([0, 0, -1])
    v_target = np.array([-x, -y, -z])  # 目标向量

    # 将目标向量归一化
    v_target_normalized = v_target / np.linalg.norm(v_target)

    # 计算旋转轴和角度
    rotation_axis = np.cross(v_original, v_target_normalized)
    angle = np.arccos(np.clip(np.dot(v_original, v_target_normalized), -1.0, 1.0))

    # 构建四元数旋转
    rotation_quaternion = R.from_rotvec(rotation_axis * angle / np.linalg.norm(rotation_axis))

    # 将四元数旋转转换为4x4旋转矩阵
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_quaternion.as_matrix()

    # 更新相机的位姿矩阵
    camera_pose_updated = rotation_matrix_4x4 @ camera_pose
    translation_matrix = trimesh.transformations.translation_matrix([x, y, z])
    combined_transform = np.dot(translation_matrix, camera_pose_updated)

    return combined_transform
