from PIL import Image
import numpy as np
import struct
# 打开图片
img = Image.open('data/test.jpg')

# 转换为numpy数组
img_array = np.array(img).astype(np.float32) / 255
width, height = img.size
channels = img_array.shape[2] if len(img_array.shape) > 2 else 1

binary_width = struct.pack('i', width)
binary_height = struct.pack('i', height)
binary_channels = struct.pack('i', channels)


with open('data/test.bin', 'wb') as f:
    f.write(binary_width)
    f.write(binary_height)
    f.write(binary_channels)
    img_array.tofile(f)