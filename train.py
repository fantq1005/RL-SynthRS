# train.py
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, defaultdict
import logging
from dqn_image_synthesis import (
    KidCalculator,
    ReplayMemory,
    DQN,
    ImageSynthesisEnv
)
import renderer  # 导入重构后的 renderer 模块


def create_run_folder(base_folder="runs"):
    # 创建 runs 目录，如果没有的话
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # 使用当前时间戳创建文件夹名称，确保文件夹唯一
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_folder, f"run_{timestamp}")

    # 创建该文件夹
    os.makedirs(run_folder)

    return run_folder


def copy_settings_to_run_folder(run_folder, settings_file="settings.txt"):
    # 如果 settings.txt 存在，将其复制到新创建的文件夹
    if os.path.exists(settings_file):
        shutil.copy(settings_file, run_folder)
    else:
        print(f"Warning: {settings_file} not found!")


def read_settings(run_folder, settings_file="settings.txt"):
    # 读取指定路径的 settings.txt 配置文件
    settings_path = os.path.join(run_folder, settings_file)
    settings = {}

    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            for line in f:
                if line.startswith('--'):
                    key, value = line.strip().split(maxsplit=1)
                    settings[key[2:]] = value  # Remove '--' from key
    else:
        print(f"Warning: {settings_file} not found in {run_folder}")

    return settings


def update_settings(run_folder, settings, settings_file="settings.txt"):
    # 更新 settings.txt 文件
    settings_path = os.path.join(run_folder, settings_file)

    with open(settings_path, 'w') as f:
        for key, value in settings.items():
            f.write(f"--{key} {value}\n")

def setup_logger(run_folder):
    log_file = os.path.join(run_folder, 'train_log.txt')  # 日志文件放在当前训练文件夹
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 以写入模式打开日志文件，每次启动训练都会覆盖
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger()

def main():
    # 设置随机种子以确保结果可重复
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_folder = create_run_folder()
    copy_settings_to_run_folder(run_folder, settings_file='settings.txt')
    settings = read_settings(run_folder)
    update_settings(run_folder, settings)
    print(f"Training settings are saved in: {run_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    logger = setup_logger(run_folder)  # 将日志记录到训练文件夹中的 train_log.txt
    logger.info("Training started...")

    # 定义超参数
    INITIAL_VALUES = {
        'tx_min': -0.18,
        'tx_max': 0.18,
        'ty_min': -0.18,
        'ty_max': 0.18,
        'camera_min_h': 0.3,
        'camera_max_h': 0.5,
        'light_I': 20.0,
        'light_min_angle': 30.0,
        'light_max_angle': 70.0
    }

    PARAM_BOUNDS = {
        'tx_min': (-0.3, 0.3),
        'tx_max': (-0.3, 0.3),
        'ty_min': (-0.3, 0.3),
        'ty_max': (-0.3, 0.3),
        'camera_min_h': (0.1, 0.8),
        'camera_max_h': (0.5, 1.0),
        'light_I': (5.0, 100.0),
        'light_min_angle': (0.0, 90.0),
        'light_max_angle': (30.0, 90.0)
    }

    PARAM_STEP_SIZES = {
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

    PARAMETERS = ['tx_min', 'tx_max', 'ty_min', 'ty_max',
                  'camera_min_h', 'camera_max_h', 'light_I',
                  'light_min_angle', 'light_max_angle']
    ACTION_SPACE = []
    for param in PARAMETERS:
        ACTION_SPACE.append((param, -1))  # 减少
        ACTION_SPACE.append((param, 1))   # 增加

    NUM_ACTIONS = len(ACTION_SPACE)  # 18
    STATE_SIZE = len(INITIAL_VALUES)  # 9
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 10
    NUM_EPISODES = 100
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 500

    # 定义真实图像文件夹路径
    real_images_folder = r"F:\bishe\bishe.7z\ultralytics-main\datasets\10perDOSOTA\images\test"  # 请替换为真实图像的路径

    # 初始化KID计算器
    logger.info("初始化KID计算器...")
    print("初始化KID计算器...")
    kid_calculator = KidCalculator(real_images_folder, device=device)
    logger.info("KID计算器初始化完成。")
    print("KID计算器初始化完成。")

    # 解析配置文件
    config_file_path = os.path.join(run_folder, "settings.txt")
    config = renderer.parse_settings_file(settings_file=config_file_path)
    image_dir = os.path.join(run_folder, config.output_folder, 'images')
    # 预加载所有车辆模型
    preloaded_models = renderer.preload_models_parallel(config.car_directory)
    if not preloaded_models:
        print("没有预加载到任何模型。请检查车辆模型文件夹路径和模型格式。")
        exit(1)

    # 初始化渲染器
    logger.info("初始化渲染器...")
    print("初始化渲染器...")
    try:
        renderer_instance = renderer.render_scenes(config, preloaded_models, run_folder)
        logger.info("渲染器初始化完成。")
        print("渲染器初始化完成。")
    except Exception as e:
        logger.error(f"渲染器初始化失败: {e}")
        print(f"渲染器初始化失败: {e}")
        exit(1)

    # 初始化环境
    logger.info("初始化环境...")
    print("初始化环境...")
    env = ImageSynthesisEnv(
        kid_calculator=kid_calculator,
        run_folder=run_folder,
        action_space=ACTION_SPACE,
        initial_values=INITIAL_VALUES,
        image_dir=image_dir,
        setting_file='settings.txt',
        param_bounds=PARAM_BOUNDS,
        step_sizes=PARAM_STEP_SIZES,  # 传递步长
        render_script='renderer.py',  # 使用新的渲染脚本
        max_steps_per_episode=64  # 设置每个回合的最大步数
    )
    logger.info("环境初始化完成。")
    print("环境初始化完成。")

    # 初始化DQN和目标网络
    policy_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
    target_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    epsilon = EPS_START

    reward_window = deque(maxlen=100)  # 平滑100回合的奖励
    action_counts = defaultdict(int)

    logger.info("开始训练...")
    print("开始训练...")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step_in_episode = 0
        max_reward_in_episode = -np.inf
        min_reward_in_episode = np.inf
        while not done:
            steps_done += 1
            step_in_episode += 1
            # 更新ε
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            # 选择动作
            if random.random() < epsilon:
                action = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            # 记录动作分布
            action_counts[action] += 1
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            # 更新最大和最小奖励
            if reward > max_reward_in_episode:
                max_reward_in_episode = reward
            if reward < min_reward_in_episode:
                min_reward_in_episode = reward
            # 存储经验
            memory.push((state, action, reward, next_state, done))
            state = next_state
            # 训练
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
                batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

                # 当前Q值
                current_q = policy_net(batch_state).gather(1, batch_action).squeeze()

                # 下一个状态的最大Q值
                with torch.no_grad():
                    max_next_q = target_net(batch_next_state).max(1)[0]

                # 计算目标Q值
                target_q = batch_reward + GAMMA * max_next_q * (1 - batch_done)

                # 计算损失
                loss = nn.MSELoss()(current_q, target_q)

                # 优化
                optimizer.zero_grad()
                loss.backward()

                # 计算梯度范数
                total_norm = 0
                for p in policy_net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                optimizer.step()

                # 记录损失和梯度范数到日志
                logger.info(f"Step {steps_done}, Loss: {loss.item():.6f}, Gradient Norm: {total_norm:.6f}")

        # 记录回合奖励
        reward_window.append(total_reward)
        average_reward = np.mean(reward_window)

        # 更新目标网络
        if (episode + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logger.info(f"目标网络已在回合 {episode + 1} 更新。")

        # 记录和打印日志
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES}, "
                    f"Total Reward: {total_reward:.4f}, "
                    f"Average Reward: {average_reward:.4f}, "
                    f"Epsilon: {epsilon:.4f}, "
                    f"Memory Size: {len(memory)}, "
                    f"Action Distribution: {dict(action_counts)}, "
                    f"Max Reward in Episode: {max_reward_in_episode:.4f}, "
                    f"Min Reward in Episode: {min_reward_in_episode:.4f}")
        print(f"Episode {episode + 1}/{NUM_EPISODES}, "
              f"Total Reward: {total_reward:.4f}, "
              f"Average Reward: {average_reward:.4f}, "
              f"Epsilon: {epsilon:.4f}, "
              f"Memory Size: {len(memory)}, "
              f"Actions Taken: {dict(action_counts)}, "
              f"Max Reward: {max_reward_in_episode:.4f}, "
              f"Min Reward: {min_reward_in_episode:.4f}")

        # 重置动作计数
        action_counts = defaultdict(int)

    model_path = os.path.join(run_folder, 'dqn_policy_net.pth')
    torch.save(policy_net.state_dict(), model_path)
    logger.info(f"训练完成，模型已保存到 {model_path}")
    print('完成')

    # 清理渲染器
    renderer_instance.delete_renderer()

if __name__ == "__main__":
    main()
