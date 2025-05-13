import numpy as np
import pyrender
from OpenGL.error import GLError
from pyrender import RenderFlags
import os
from utils.scene_creation import add_model_to_scene, add_background_to_scene, add_lights
from utils.model_transforms import get_random_camera_pose, get_random_transform
from utils.image_processing import get_bbox, apply_feather_to_foreground_edges, harmonization
from utils.utils import get_random_background, bbox_intersects
import imageio
from tqdm import tqdm
import cv2
import random
import configargparse
import trimesh
from multiprocessing import Pool, cpu_count
import sys

def parse_settings_file(settings_file=None):
    parser = configargparse.ArgParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='Config file path')
    parser.add_argument('--output_folder', type=str, default='output_cars', help="输出文件夹")
    parser.add_argument('--n', type=int, default=15000, help="渲染的数量")
    parser.add_argument('--bak_directory', type=str, default=r'F:/bishe/bishe.7z/hecheng/background/crop2048obj',
                       help="背景文件夹路径")
    parser.add_argument('--car_directory', type=str, default=r'F:/bishe/bishe.7z/hecheng/object/shapenet_cars',
                       help="车辆模型文件夹路径")
    parser.add_argument('--image_width', type=int, default=512, help="渲染图像的宽度")
    parser.add_argument('--image_height', type=int, default=512, help="渲染图像的高度")
    parser.add_argument('--max_car_num', type=int, default=3, help="最大车辆数")
    parser.add_argument('--min_car_num', type=int, default=1, help="最小车辆数")
    parser.add_argument('--tx_min', type=float, default=-0.15, help="随机位置偏移tx的最小值")
    parser.add_argument('--tx_max', type=float, default=0.15, help="随机位置偏移tx的最大值")
    parser.add_argument('--ty_min', type=float, default=-0.15, help="随机位置偏移ty的最小值")
    parser.add_argument('--ty_max', type=float, default=0.15, help="随机位置偏移ty的最大值")
    parser.add_argument('--tz', type=float, default=0.002, help="随机位置偏移tz的值")
    parser.add_argument('--scale_range_min', type=float, default=0.04, help="缩放因子的最小值")
    parser.add_argument('--scale_range_max', type=float, default=0.06, help="缩放因子的最大值")
    parser.add_argument('--rotation_x_angle', type=float, default=90, help="绕X轴的旋转角度")
    parser.add_argument('--camera_min_h', type=float, default=0.3, help="相机高度最小值")
    parser.add_argument('--camera_max_h', type=float, default=0.5, help="相机高度最大值")
    parser.add_argument('--light_I', type=float, default=20, help="光强")
    parser.add_argument('--light_min_angle', type=float, default=30, help="光照旋转角最小值")
    parser.add_argument('--light_max_angle', type=float, default=70, help="光照旋转角最大值")

    args = []
    if settings_file:
        args.extend(['--config', settings_file])

    config = parser.parse_args(args)
    return config

def load_single_model(model_path):
    """
    加载单个模型并转换为pyrender.Mesh。

    Args:
        model_path (str): 模型文件路径。

    Returns:
        pyrender.Mesh 或 None: 转换后的pyrender.Mesh对象，或在失败时返回None。
    """
    try:
        mesh = trimesh.load(model_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            # 如果是场景（包含多个网格），合并为单一网格
            mesh = trimesh.util.concatenate(mesh.dump())
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        return pyrender_mesh
    except Exception as e:
        print(f"加载模型失败 {model_path}: {e}")
        return None

def preload_models_parallel(car_directory, supported_extensions=('.obj', '.stl', '.glb', '.gltf'), max_workers=None):
    """
    并行预加载所有车辆模型到内存中，并显示加载进度。

    Args:
        car_directory (str): 车辆模型文件夹路径。
        supported_extensions (tuple): 支持的模型文件扩展名。
        max_workers (int, optional): 并行工作的最大进程数。默认为CPU核心数。

    Returns:
        list[pyrender.Mesh]: 预加载的车辆模型列表。
    """
    preloaded_models = []
    model_files = []

    # 收集所有支持的模型文件路径
    for root, _, files in os.walk(car_directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                model_path = os.path.join(root, file)
                model_files.append(model_path)

    print(f"开始预加载 {len(model_files)} 个模型...")

    # 设置最大进程数
    if max_workers is None:
        max_workers = cpu_count()

    # 使用进程池并行加载模型
    with Pool(processes=max_workers) as pool:
        # 使用imap_unordered配合tqdm显示进度条
        for pyrender_mesh in tqdm(pool.imap_unordered(load_single_model, model_files), total=len(model_files), desc="加载模型", unit="模型"):
            if pyrender_mesh is not None:
                preloaded_models.append(pyrender_mesh)

    print(f"成功预加载了 {len(preloaded_models)} 个模型。")
    return preloaded_models

def render_scenes(config, preloaded_models, run_folder):
    # 创建输出文件夹
    output_folder = os.path.join(run_folder, config.output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_folder = os.path.join(output_folder, "images")
    bbox_folder = os.path.join(output_folder, "bbox_txt")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(bbox_folder, exist_ok=True)

    # 初始化渲染器（在循环外部）
    try:
        renderer = pyrender.OffscreenRenderer(config.image_width, config.image_height)
    except Exception as e:
        print(f"无法初始化 OffscreenRenderer: {e}")
        return

    try:
        # 渲染过程
        for i in tqdm(range(config.n), desc="渲染场景", unit="场景"):
            try:
                # 从配置文件读取路径
                background_path = get_random_background(config.bak_directory)
                bbox_file = os.path.join(bbox_folder, f'rendered_scene_{i}.txt')
                image_file = os.path.join(images_folder, f'rendered_scene_{i}.png')

                # 初始化场景和摄像机
                scene = pyrender.Scene()
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                h = random.uniform(config.camera_min_h, config.camera_max_h)
                camera_pose = get_random_camera_pose(height=h)
                cumulative_mask = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
                m = random.randint(config.min_car_num, config.max_car_num)
                added_bboxes = []

                # 渲染每个车辆模型
                for j in range(m):
                    # 随机选择预加载的模型
                    model = random.choice(preloaded_models)
                    transform_matrix = get_random_transform(
                        tx_range=(config.tx_min, config.tx_max),
                        ty_range=(config.ty_min, config.ty_max),
                        tz=config.tz,
                        rotation_x_angle=config.rotation_x_angle,
                        rotation_z_range=(0, 360),
                        scale_range=(config.scale_range_min, config.scale_range_max)
                    )

                    # 使用临时场景进行渲染以获取mask和bbox
                    temp_scene = pyrender.Scene()
                    temp_scene.add(camera, pose=camera_pose)
                    temp_scene.add(model, pose=transform_matrix)

                    # 使用复用的渲染器进行渲染
                    color, depth = renderer.render(temp_scene)

                    # 生成二进制mask
                    binary_mask = np.where(depth > 0, 255, 0).astype(np.uint8)
                    try:
                        bbox = get_bbox(depth)
                    except IndexError:
                        print(f"No valid bbox found for scene {i}, car {j}. Skipping this car.")
                        continue  # 跳过当前车辆，继续下一个

                    overlap = False

                    for existing_bbox in added_bboxes:
                        if bbox_intersects(bbox, existing_bbox):
                            overlap = True
                            break

                    if overlap:
                        continue

                    cumulative_mask = cv2.bitwise_or(cumulative_mask, binary_mask)
                    added_bboxes.append(bbox)
                    scene.add(model, pose=transform_matrix)

                cumulative_mask = cumulative_mask.astype(np.uint8)

                # 加载背景并渲染场景
                add_background_to_scene(background_path, scene)
                scene.add(camera, pose=camera_pose)
                add_lights(scene, I1=0, I2=config.light_I, min_angle=config.light_min_angle, max_angle=config.light_max_angle)

                # 使用复用的渲染器进行最终渲染
                color, depth = renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
                color = apply_feather_to_foreground_edges(color, cumulative_mask, edge_width=1, blur_amount=(1, 1))
                color = harmonization(color, cumulative_mask, color_correction=0.2)
                imageio.imwrite(image_file, color)

                # 写入label文件，覆盖原有内容
                with open(bbox_file, 'w') as file:
                    for bbox in added_bboxes:
                        # 计算类别、中心点、宽度和高度
                        # 假设类别为0（你可以根据需要调整）
                        x_center = (bbox[0] + bbox[2]) / 2.0 / config.image_width
                        y_center = (bbox[1] + bbox[3]) / 2.0 / config.image_height
                        width = (bbox[2] - bbox[0]) / config.image_width
                        height = (bbox[3] - bbox[1]) / config.image_height

                        # 写入label文件：类别 x_center y_center width height
                        file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            except GLError as e:
                print(f"OpenGL error, skipping iteration {i}: {e}")
                continue

            except ValueError as e:
                print(f"ValueError at iteration {i}, skipping: {e}")
                continue

            except TypeError as e:
                print(f"TypeError at iteration {i}, skipping: {e}")
                continue

            except RuntimeError as e:
                print(f"遇到了一个错误在迭代 {i}: {e}")
                continue

    finally:
        renderer.delete()  # 在渲染完成后删除渲染器

if __name__ == "__main__":
    config_file_path = sys.argv[1]  # 获取命令行参数中的文件路径
    # config_file_path = "settings.txt"
    print(f"使用的配置文件路径: {config_file_path}")  # 打印配置文件路径，确认是否正确
    # 检查配置文件是否存在
    if config_file_path and not os.path.isfile(config_file_path):
        print(f"配置文件不存在: {config_file_path}")
        exit(1)
    # 解析配置
    config = parse_settings_file(settings_file=config_file_path)
    print("配置文件解析完成.")  # 打印是否成功解析配置
    run_folder = os.path.dirname(config_file_path)
    # 并行预加载所有车辆模型
    preloaded_models = preload_models_parallel(config.car_directory)
    if not preloaded_models:
        print("没有预加载到任何模型。请检查车辆模型文件夹路径和模型格式。")
        exit(1)
    # 开始渲染
    render_scenes(config, preloaded_models, run_folder)