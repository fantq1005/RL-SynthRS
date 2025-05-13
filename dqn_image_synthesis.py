# dqn_image_synthesis.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import numpy as np
import torch
import torch.nn as nn
from collections import deque, defaultdict
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import time
import random




class KidCalculator:
    """
    用于计算KID的类。与FidCalculator类似，只不过把Fid相关计算改为KID。
    """
    def __init__(self, real_images_folder, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_inception_model()
        self.transform = self.get_transform()
        # 载入真实图像
        self.real_images = self.load_images_from_folder(real_images_folder)
        if not self.real_images:
            raise ValueError(f"未找到真实图像在文件夹: {real_images_folder}")
        print("正在提取真实图像特征...")
        # 提取真实图像的Inception特征
        self.real_activations = self.get_activations(self.real_images)
        print("真实图像特征提取完成。")

    def load_inception_model(self):
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        # 去掉最后的分类层，只保留特征部分
        model.fc = nn.Identity()
        model.eval()
        model.to(self.device)
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def load_images_from_folder(self, folder_path):
        images = []
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in supported_formats):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"无法加载图像 {img_path}: {e}")
        return images

    def get_activations(self, images, batch_size=16):
        """
        获取一批图像的Inception激活（全连接层输入前或global pool后的特征）。
        """
        self.model.eval()
        activations = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch = torch.stack([self.transform(img) for img in batch]).to(self.device)
                preds = self.model(batch)
                if preds.ndimension() == 4:
                    preds = adaptive_avg_pool2d(preds, output_size=(1, 1)).squeeze()
                activations.append(preds.cpu())
        return torch.cat(activations, dim=0)

    def compute_kid(self, generated_images_folder):
        """
        计算KID值：值越小越好。
        """
        gen_images = self.load_images_from_folder(generated_images_folder)
        if not gen_images:
            print(f"未找到生成图像在文件夹: {generated_images_folder}")
            return np.inf

        print("正在提取生成图像特征...")
        gen_activations = self.get_activations(gen_images)
        print("开始计算KID...")
        kid_score = self._calculate_kid(self.real_activations, gen_activations)
        return kid_score

    def _calculate_kid(self, real_acts, gen_acts, num_subsets=10, subset_size=1000):
        """
        利用多次采样近似来估计KID。可以用多次随机抽取子集的MMD平均值。
        这里采用一个典型的 polynomial kernel 方案:
            k(x, y) = ( x^T y / d + 1 )^3
        如果数据很多，可以适当提高num_subsets或subset_size，或根据需要做改进。
        """
        real_acts = real_acts.numpy()
        gen_acts = gen_acts.numpy()
        kid_values = []

        # 保证每次抽取子集大小不超过真实或生成特征数
        m = min(len(real_acts), len(gen_acts), subset_size)

        for _ in range(num_subsets):
            real_idx = np.random.choice(len(real_acts), m, replace=False)
            gen_idx = np.random.choice(len(gen_acts), m, replace=False)
            x = real_acts[real_idx]
            y = gen_acts[gen_idx]

            kid_values.append(self._polynomial_mmd_averages(x, y))

        return np.mean(kid_values)

    def _polynomial_mmd_averages(self, x, y):
        """
        计算单次抽样的 polynomial MMD^2 (无偏估计)。
        polynomial kernel: k(a, b) = ( (a·b)/d + 1 )^3
        """
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        x_norm = (x**2).sum(1)
        y_norm = (y**2).sum(1)

        # 先计算点积矩阵
        k_xx = self._polynomial_kernel(x, x, x_norm, x_norm)
        k_yy = self._polynomial_kernel(y, y, y_norm, y_norm)
        k_xy = self._polynomial_kernel(x, y, x_norm, y_norm)

        return self._mmd2_unbiased(k_xx, k_yy, k_xy)

    def _polynomial_kernel(self, x, y, x_norm, y_norm, degree=3):
        """
        实现 polynomial kernel: k(a, b) = ( (a·b)/d + 1 )^3
        其中 d 是特征维度，用于归一化点积。
        """
        d = x.shape[1]
        # 矩阵乘法得到所有 (a·b)
        prod = np.dot(x, y.T) / d
        # 加 1 后立方
        return (prod + 1) ** degree

    def _mmd2_unbiased(self, Kxx, Kyy, Kxy):
        """
        计算无偏MMD^2。
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        这里用unbiased估计，即对角项不计。
        """
        m = Kxx.shape[0]  # x与y子集大小相等
        # 去掉对角项
        diag_x = np.diagonal(Kxx)
        diag_y = np.diagonal(Kyy)

        sum_xx = (Kxx.sum() - diag_x.sum()) / (m * (m-1))
        sum_yy = (Kyy.sum() - diag_y.sum()) / (m * (m-1))
        sum_xy = Kxy.sum() / (m * m)

        return sum_xx + sum_yy - 2 * sum_xy


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class ImageSynthesisEnv:
    def __init__(self, kid_calculator, run_folder, action_space, initial_values=None,
                 param_bounds=None, render_script='render_random_v4.py',
                 setting_file='settings.txt', image_dir='generated_images/images',
                 max_steps_per_episode=100, step_sizes=None):
        """
        初始化环境

        :param fid_calculator: FidCalculator实例
        :param action_space: 动作空间列表，每个动作为(tuple) (parameter, delta)
        :param initial_values: 参数的初始值字典
        :param param_bounds: 参数的边界字典，格式为 {parameter: (min, max)}
        :param render_script: 渲染脚本路径
        :param setting_file: 基础设置文件路径
        :param image_dir: 生成图像目录
        :param max_steps_per_episode: 每个回合的最大步数
        :param step_sizes: 每个参数的调整步长字典
        """
        self.kid_calculator = kid_calculator
        self.action_space = action_space
        self.render_script = render_script
        self.base_setting_file = setting_file
        self.image_dir = image_dir
        self.max_steps_per_episode = max_steps_per_episode
        self.run_folder = run_folder

        # 定义参数及其初始值
        default_initial = {
            'tx_min': -0.18,  # 允许为负值
            'tx_max': 0.18,
            'ty_min': -0.18,  # 允许为负值
            'ty_max': 0.18,
            'camera_min_h': 0.3,
            'camera_max_h': 0.5,
            'light_I': 20.0,
            'light_min_angle': 30.0,
            'light_max_angle': 70.0
        }
        self.params = initial_values if initial_values else default_initial.copy()

        # 定义参数边界
        default_bounds = {
            'tx_min': (-0.5, -0.3),
            'tx_max': (0.3, 0.5),
            'ty_min': (-0.5, -0.3),
            'ty_max': (0.3, 0.5),
            'camera_min_h': (0.1, 0.8),
            'camera_max_h': (0.5, 1.0),
            'light_I': (5.0, 100.0),
            'light_min_angle': (0.0, 90.0),
            'light_max_angle': (30.0, 90.0)
        }
        self.param_bounds = param_bounds if param_bounds else default_bounds.copy()

        # 定义每个参数的步长
        default_step_sizes = {
            'tx_min': 0.01,
            'tx_max': 0.01,
            'ty_min': 0.01,
            'ty_max': 0.01,
            'camera_min_h': 0.01,
            'camera_max_h': 0.01,
            'light_I': 0.5,
            'light_min_angle': 1.0,
            'light_max_angle': 1.0
        }
        self.step_sizes = step_sizes if step_sizes else default_step_sizes.copy()

        self.step_count = 0  # 用于命名新的setting文件

    def reset(self):
        """
        重置环境到初始状态
        """
        self.step_count = 0
        self._render(step=self.step_count)
        return self._get_state()

    def step(self, action):
        """
        执行动作

        :param action: 动作索引
        :return: (next_state, reward, done)
        """
        parameter, delta = self.action_space[action]
        # 计算新的参数值
        new_value = self.params[parameter] + delta * self.step_sizes[parameter]

        # 获取参数边界
        min_bound, max_bound = self.param_bounds[parameter]

        # 确保新值在边界内
        new_value = np.clip(new_value, min_bound, max_bound)

        # 更新参数
        self.params[parameter] = new_value

        # 处理 min < max 的约束
        self._enforce_constraints(parameter)

        self.step_count += 1
        self._render(step=self.step_count)
        kid = self.kid_calculator.compute_kid(self.image_dir)
        reward = self.compute_reward(kid)

        # 定义回合结束条件
        done = self.step_count >= self.max_steps_per_episode
        next_state = self._get_state()
        return next_state, reward, done

    def _enforce_constraints(self, updated_param):
        """
        强制执行参数之间的约束关系

        :param updated_param: 最近更新的参数名
        """
        # 定义参数之间的关系
        relations = {
            'tx_min': 'tx_max',
            'ty_min': 'ty_max',
            'camera_min_h': 'camera_max_h',
            'light_min_angle': 'light_max_angle'
        }

        if updated_param in relations:
            max_param = relations[updated_param]
            # 确保 min < max
            if self.params[updated_param] >= self.params[max_param]:
                self.params[max_param] = self.params[updated_param] + self.step_sizes
                # 确保 max 不超过其边界
                self.params[max_param] = np.clip(self.params[max_param],
                                                self.param_bounds[max_param][0],
                                                self.param_bounds[max_param][1])

        # 反向关系：如果 max 被更新，确保 min < max
        reverse_relations = {v: k for k, v in relations.items()}
        if updated_param in reverse_relations:
            min_param = reverse_relations[updated_param]
            if self.params[min_param] >= self.params[updated_param]:
                self.params[min_param] = self.params[updated_param] - self.step_sizes
                # 确保 min 不低于其边界
                self.params[min_param] = np.clip(self.params[min_param],
                                                self.param_bounds[min_param][0],
                                                self.param_bounds[min_param][1])

    def _render(self, step):
        """
        渲染当前环境，更新settings.txt并生成step-specific的setting文件。
        """
        # 获取当前训练文件夹路径
        run_folder = self.run_folder  # 假设 run_folder 是传递给类的一个参数

        # 更新setting.txt文件的路径
        base_setting_file_path = os.path.join(run_folder, self.base_setting_file)

        # 读取原始settings.txt内容到字典中
        settings = {}
        if os.path.exists(base_setting_file_path):
            with open(base_setting_file_path, 'r') as f:
                for line in f:
                    if line.startswith('--'):
                        key, value = line.strip().split(maxsplit=1)
                        settings[key[2:]] = value  # 移除 '--' 前缀，作为字典的key
        else:
            print(f"未找到设置文件: {base_setting_file_path}")

        # 更新指定key的value，例：更新 tx_max
        for key, value in self.params.items():
            if key in settings:
                settings[key] = str(value)  # 更新已有的key
            else:
                settings[key] = str(value)  # 如果是新增的key，则添加

        # 更新原始setting.txt
        with open(base_setting_file_path, 'w') as f:
            for key, value in settings.items():
                f.write(f"--{key} {value}\n")

        print(f"更新设置文件: {base_setting_file_path}")

        # 生成step-specific的settings文件的路径
        step_setting_file_path = os.path.join(run_folder, f"settings_step_{step}.txt")
        with open(step_setting_file_path, 'w') as f:
            for key, value in settings.items():
                f.write(f"--{key} {value}\n")

        print(f"生成步骤文件: {step_setting_file_path}")

        # 构建渲染脚本命令
        cmd = ['python', self.render_script, base_setting_file_path]

        print(f"调用渲染脚本: {' '.join(cmd)}")

        # 调用渲染脚本生成图像
        # try:
        subprocess.run(cmd, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"渲染脚本执行失败: {e}")
        #     raise e

        print(f"渲染完成，步骤 {step} 使用了配置文件: {base_setting_file_path}")

    def compute_reward(self, kid):
        """
        根据 FID 计算奖励，FID 越低，奖励越高

        :param fid: FID 值
        :return: 奖励值
        """
        if kid == np.inf:
            return -1  # 极差的情况下给予负奖励
        reward = -kid  # 负 FID 作为奖励
        return reward

    def _get_state(self):
        """
        获取当前状态

        :return: numpy数组，包含所有参数的当前值
        """
        state = np.array([
            self.params['tx_min'],
            self.params['tx_max'],
            self.params['ty_min'],
            self.params['ty_max'],
            self.params['camera_min_h'],
            self.params['camera_max_h'],
            self.params['light_I'],
            self.params['light_min_angle'],
            self.params['light_max_angle']
        ], dtype=np.float32)
        return state

