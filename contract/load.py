from PIL import Image
import numpy as np

# 图像路径
img_path = 'img_1.png'

# 打开图像（PIL 自动保持原始模式，如 RGB、RGBA、L 等）
img = Image.open(img_path)

# 转为 numpy 数组
img_array = np.array(img)

print(img_array.shape)