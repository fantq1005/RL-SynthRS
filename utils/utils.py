import random
import os
import glob

def generate_random_efficient():
    if random.choice([True, False]):  # 随机选择生成的区间
        return random.uniform(-70, -30)
    else:
        return random.uniform(30, 70)

def get_random_background(directory):
    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    if not subdirs:
        return None
    selected_subdir = random.choice(subdirs)
    obj_files = glob.glob(os.path.join(selected_subdir, "*.obj"))
    if not obj_files:
        return None
    return random.choice(obj_files)

def get_random_obj(directory):
    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
    if not subdirs:
        return None
    selected_subdir = random.choice(subdirs)
    models_dir = os.path.join(selected_subdir, "models")  # 进入models子目录
    obj_files = glob.glob(os.path.join(models_dir, "*.obj"))
    if not obj_files:
        return None
    return random.choice(obj_files)

def bbox_intersects(bbox1, bbox2):
    """
    判断两个包围框是否相交
    bbox 格式: (x_min, y_min, x_max, y_max)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 检查水平和垂直方向上是否有交集
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)
