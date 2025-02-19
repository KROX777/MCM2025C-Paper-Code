from PIL import Image, ImageFilter
import numpy as np

# 打开图片
image_path = "Screenshot 2025-01-27 at 16.18.19.png"
img = Image.open(image_path)

# 转换为 NumPy 数组
img_array = np.array(img)

# 获取图片尺寸
height, width, channels = img_array.shape

# 创建一个从中心线开始的渐变遮罩
gradient_height = height // 2  # 从中心线到底部的高度
gradient = np.linspace(0, 1, gradient_height)  # 渐变从 0 到 1
gradient = np.concatenate((np.zeros(height - gradient_height), gradient))[:, None]  # 顶部区域保持 0

# 扩展遮罩到图片宽度
mask = np.repeat(gradient, width, axis=1)

# 如果图片有 Alpha 通道，扩展 mask 到 4 个通道
if channels == 4:
    mask = np.repeat(mask[:, :, None], 4, axis=2)
else:
    mask = np.repeat(mask[:, :, None], 3, axis=2)

# 模糊整张图片
blurred_img = img.filter(ImageFilter.GaussianBlur(radius=15))
blurred_array = np.array(blurred_img)

# 应用渐变模糊（线性插值）
result_array = img_array * (1 - mask) + blurred_array * mask
result_array = result_array.astype(np.uint8)

# 转回 Pillow 图像
result_img = Image.fromarray(result_array)

# 保存结果
result_img.save("output_gradient_blur_center.png")
result_img.show()