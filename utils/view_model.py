import os
import shutil


def copy_screenshots(src_folder, dest_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            if dir_name == 'screenshots':  # 找到名为'screenshots'的文件夹
                screenshots_path = os.path.join(root, dir_name)

                # 获取screenshots文件夹中的所有文件
                for file_name in os.listdir(screenshots_path):
                    file_path = os.path.join(screenshots_path, file_name)

                    # 如果是文件，则复制到目标文件夹
                    if os.path.isfile(file_path):
                        shutil.copy(file_path, dest_folder)
                        print(f"已复制 {file_path} 到 {dest_folder}")


# 使用示例
# src_folder = r'F:/dataset/ShapeNetCore.v2/ShapeNetCore.v2/02691156'  # 飞机
src_folder = r'F:/dataset/ShapeNetCore.v2/ShapeNetCore.v2/02958343'  # 车辆
dest_folder = 'screenshots_car'  # 替换为你要复制到的目标文件夹路径

copy_screenshots(src_folder, dest_folder)
