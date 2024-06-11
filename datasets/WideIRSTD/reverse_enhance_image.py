import os
from PIL import Image,ImageFilter
import os
import numpy as np
from PIL import Image
from skimage import exposure, filters
from skimage.filters import gaussian

# 数据集路径和保存处理后图像的路径
dataset_dir = "./image_raw"
output_dir = "./image_reverse"

# 确保保存处理后图像的路径存在
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集中的图像
for filename in os.listdir(dataset_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # 读取图像
        image_path = os.path.join(dataset_dir, filename)
        image = Image.open(image_path)

        # 将图像转换为RGB模式（如果不是RGB格式的图像）
        image = image.convert("RGB")
        image = Image.eval(image, lambda x: 255 - x)
        # 创建一个与图像尺寸相同的纯白图像
        white_image = Image.new("RGB", image.size, (255, 255, 255))

        # 逐元素相减
        subtracted_image = Image.blend(image, white_image, alpha=0.2)
        
        # 图像增强算法（这里仅作示例，你可以根据实际需求进行更改）
        enhanced_image = subtracted_image.filter(ImageFilter.SHARPEN)

        # 保存处理后的图像
        output_path = os.path.join(output_dir, filename)
        enhanced_image.save(output_path)

        # 可选：显示处理后的图像
