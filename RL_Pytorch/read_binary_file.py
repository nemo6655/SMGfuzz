import struct
import numpy as np
import os
import codecs
   
# 文件夹路径
train_path = "output/train"
seed_path = "output/queue"

# 获取文件夹中的所有文件名
train_names = os.listdir(train_path)
seed_names = os.listdir(seed_path)
# 遍历文件名
for file_name in train_names:
    # 检查文件名开头是否为 "bitmap"
    if file_name.startswith("bitmap"):
        # 构建文件的完整路径
        file_path = os.path.join(train_path, file_name)
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            # 将二进制数据转换为 numpy 数组
            matrix = np.frombuffer(binary_data, dtype=np.uint8)
            # 输出矩阵数据
            matrix = matrix.reshape((256, 256))
            np.savetxt('Decode_Data/bitmap/'+file_name+'.txt', matrix, fmt='%d')
            print(file_name+'已解析')
    if file_name.startswith("statemap"):
        # 构建文件的完整路径
        file_path = os.path.join(train_path, file_name)
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            # 将二进制数据转换为 numpy 数组
            matrix = np.frombuffer(binary_data, dtype=np.uint8)
            # 输出矩阵数据
            matrix = matrix.reshape((256, 256))
            np.savetxt('Decode_Data/statemap/'+file_name+'.txt', matrix, fmt='%d')
            print(file_name+'已解析')

for file_name in seed_names:
    if file_name.startswith("id"):
        file_path = os.path.join(seed_path, file_name)
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            decoded_text = codecs.decode(binary_data, 'ascii', 'ignore')

        with open('Decode_Data/seed/'+file_name+'.txt', 'w', encoding='ascii') as file:
            file.write(decoded_text)
        print(file_name+'已解析')



