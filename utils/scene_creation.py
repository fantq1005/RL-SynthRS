import trimesh
import pyrender
import numpy as np
import random


def add_vehicle_random_position(scene, model_path):
    # 随机位置偏移
    tx = random.uniform(-0.1, 0.1)
    ty = random.uniform(-0.1, 0.1)
    tz = 0.001
    # 旋转和缩放变换矩阵
    rotation_matrix_x = trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0], [0, 0, 0])
    translation_matrix = trimesh.transformations.translation_matrix([tx, ty, tz])
    scale_factor = 0.05  # 缩放到原来的0.02倍大小
    scale_matrix = np.eye(4) * scale_factor
    scale_matrix[3, 3] = 1  # 保持齐次坐标不变
    rotation_matrix_z = trimesh.transformations.rotation_matrix(
        np.radians(random.uniform(0, 360)), [0, 0, 1], [0, 0, 0])
    # 组合变换矩阵
    combined_transform = np.dot(translation_matrix,
                                np.dot(rotation_matrix_z,
                                       np.dot(scale_matrix, rotation_matrix_x)))
    # 加载模型并应用变换
    if model_path:
        vehicle_scene = trimesh.load(model_path)

        # 应用变换并添加到场景中
        if isinstance(vehicle_scene, trimesh.Scene):
            for geom_id, geom in vehicle_scene.geometry.items():
                transformed_geom = geom.apply_transform(combined_transform)
                mesh = pyrender.Mesh.from_trimesh(transformed_geom)
                scene.add(mesh)
        else:
            transformed_geom = vehicle_scene.apply_transform(combined_transform)
            mesh = pyrender.Mesh.from_trimesh(transformed_geom)
            scene.add(mesh)

def add_lights(scene, I1=0, I2=20, min_angle=30, max_angle=70):
    # 创建光源
    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=I1)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=I2)
    # 随机旋转变换
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(generate_random_efficient(min_angle, max_angle)), [0, 1, 0], [0, 0, 0])
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(generate_random_efficient(min_angle, max_angle)), [1, 0, 0], [0, 0, 0])
    # 光源的初始位置和姿态
    light_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],  # 调整这里可以设置光源离物体的距离
        [0, 0, 0, 1]
    ])
    # 添加第一个光源（未旋转）到场景
    scene.add(light1, pose=light_pose)
    # 应用旋转变换到第二个光源的姿态
    light_pose = np.dot(rotation_matrix_y, light_pose)
    light_pose = np.dot(rotation_matrix_x, light_pose)
    # 添加第二个光源（已旋转）到场景
    scene.add(light2, pose=light_pose)

def add_model_to_scene(model_path, scene, combined_transform):
    if model_path:
        vehicle_scene = trimesh.load(model_path)
        # 如果加载的是一个场景
        if isinstance(vehicle_scene, trimesh.Scene):
            for geom_id, geom in vehicle_scene.geometry.items():
                transformed_geom = geom.apply_transform(combined_transform)
                mesh = pyrender.Mesh.from_trimesh(transformed_geom)
                scene.add(mesh)
        # 如果加载的不是场景，而是单个几何体
        else:
            transformed_geom = vehicle_scene.apply_transform(combined_transform)
            mesh = pyrender.Mesh.from_trimesh(transformed_geom)
            scene.add(mesh)

def add_background_to_scene(background_path, scene):
    if background_path:
        trimesh_obj = trimesh.load(background_path)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        scene.add(mesh)

def generate_random_efficient(a,b):
    if random.choice([True, False]):  # 随机选择生成的区间
        return random.uniform(-b, -a)
    else:
        return random.uniform(a, b)