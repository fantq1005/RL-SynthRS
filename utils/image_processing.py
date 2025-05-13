import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# def get_bbox(depth_image, threshold):
#     _, binary_image = cv2.threshold(depth_image, threshold, 255, cv2.THRESH_BINARY)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8), 8, cv2.CV_32S)
#     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     x, y, width, height, area = stats[largest_label]
#     x1, y1, x2, y2 = x, y, x + width, y + height
#     return (x1, y1, x2, y2)

def get_bbox(binary_mask):
    # 获取非零值的位置
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    # 获取包围框的最小和最大坐标
    y1, y2 = np.where(rows)[0][[0, -1]]  # 行的最小和最大值
    x1, x2 = np.where(cols)[0][[0, -1]]  # 列的最小和最大值
    return (x1, y1, x2, y2)

def plot_depth_histogram(depth_image):
    # 读取深度图
    # 检查图像是否为16位深度图，如果是，则转换到0-255范围
    if depth_image.dtype == np.uint16:
        # 假设深度图中的最大深度值为10000（可以根据实际情况调整）
        depth_image = np.clip(depth_image, 0, 10000)
        depth_image = (depth_image / 10000 * 255).astype(np.uint8)
    # 计算直方图
    # 参数256指定bin的数量，即直方图中条形的数量
    # 参数[0, 256]指定了直方图计算的区间
    histogram, bin_edges = np.histogram(depth_image, bins=256, range=(0, 256))
    # 绘制直方图
    plt.figure()
    plt.title("Depth Histogram")
    plt.xlabel("Depth value")
    plt.ylabel("Pixel count")
    plt.xlim([0, 10])
    plt.plot(bin_edges[0:-1], histogram)  # bin_edges比histogram多一个边界值，所以去掉最后一个
    plt.show()

def harmonization(img, mask, color_correction=0.2):
    img = img.astype(np.float32) / 255.
    mask = mask.astype(np.float32) / 255.
    img_fg, img_bg = extract_foreground_background(img, mask)
    matched = match_histograms(img_fg, img_bg, channel_axis=-1)
    img_fg = color_correction*matched + (1-color_correction)*img_fg
    img_fg = noise_and_blur(img_fg)
    img_composite = img.copy()
    img_composite[mask > 0] = img_fg[mask > 0]
    img_composite = (img_composite * 255.).astype(np.uint8)
    return img_composite

def noise_and_blur(img_fg):

    # forgrund blur
    img_fg = cv2.blur(img_fg, ksize=(2,2))
    img_fg = img_fg + 0.02*np.random.randn(*img_fg.shape)
    img_fg[img_fg < 0] = 0
    img_fg[img_fg > 1.0] = 1.0

    return img_fg

def extract_foreground_background(img, mask):
    # 确保掩膜是布尔类型
    mask_bool = mask.astype(bool)
    # 将掩膜应用于图像以提取前景
    foreground = np.zeros_like(img)
    foreground[mask_bool] = img[mask_bool]
    # 使用相反的掩膜提取背景
    background = np.zeros_like(img)
    background[~mask_bool] = img[~mask_bool]
    return foreground, background

def apply_feather_to_foreground_edges(color_image, binary_mask, edge_width=7, blur_amount=(9, 9)):
    binary_mask = binary_mask.astype(np.uint8)
    # 扩展掩膜边缘
    kernel = np.ones((edge_width, edge_width), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    edge_mask = cv2.subtract(dilated_mask, eroded_mask)
    # 在边缘区域应用高斯模糊
    blurred_image = cv2.GaussianBlur(color_image, blur_amount, 0)
    # 使用边缘掩膜合并模糊的图像和原图，仅在边缘区域应用模糊
    edge_blurred_image = np.where(edge_mask[..., None] == 255, blurred_image, color_image)
    # 最终图像中包含模糊的边缘，而前景和背景的其余部分保持不变
    return edge_blurred_image