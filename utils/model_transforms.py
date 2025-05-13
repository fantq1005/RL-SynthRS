import trimesh
import pyrender
import numpy as np
import random


def get_random_transform(tx_range=(-0.15, 0.15),
                         ty_range=(-0.15, 0.15),
                         tz=0.002,
                         rotation_x_angle=90,
                         rotation_z_range=(0, 360),
                         scale_range=(0.04, 0.06)):
    # 随机位置偏移
    tx = random.uniform(*tx_range)
    ty = random.uniform(*ty_range)
    # 位置偏移的z轴不随机，使用输入值tz
    # tz = 0.002

    # 旋转和缩放变换矩阵
    rotation_matrix_x = trimesh.transformations.rotation_matrix(
        np.radians(rotation_x_angle), [1, 0, 0], [0, 0, 0])

    translation_matrix = trimesh.transformations.translation_matrix([tx, ty, tz])

    scale_factor = random.uniform(*scale_range)  # 缩放
    scale_matrix = np.eye(4) * scale_factor
    scale_matrix[3, 3] = 1  # 保持齐次坐标不变

    rotation_matrix_z = trimesh.transformations.rotation_matrix(
        np.radians(random.uniform(*rotation_z_range)), [0, 0, 1], [0, 0, 0])

    # 组合变换矩阵
    combined_transform = np.dot(translation_matrix,
                                np.dot(rotation_matrix_z,
                                       np.dot(scale_matrix, rotation_matrix_x)))
    return combined_transform

def get_random_camera_pose(height=0.3):
    # 创建相机
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(random.uniform(-10, 10)), [0, 1, 0], [0, 0, 0])
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(random.uniform(-10, 10)), [1, 0, 0], [0, 0, 0])
    # 定义相机的位置和姿态
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, height],  # 调整这里来设置相机离物体的距离
        [0, 0, 0, 1]
    ])
    camera_pose= np.dot(rotation_matrix_y, camera_pose)
    camera_pose = np.dot(rotation_matrix_x, camera_pose)
    # 将相机和其姿态添加到场景中
    return camera_pose
