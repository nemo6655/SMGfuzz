import struct
import numpy as np
import os
import codecs
import sys
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def convert_to_hex(binary_data):
    hex_data = ""
    for i in range(0, 64, 4):
        hex_digit = hex((binary_data >> i) & 0xF)[2:]
        hex_data += hex_digit.upper()
    return hex_data

def generate_matrix(binary_data):
    result = []
    for data in binary_data:
        count = sum([1 for bit in bin(data)[2:] if bit == '0' or bit == 'f'])
        result.append(0 if count >= 14 else 1)
    matrix = [result[i:i+32] for i in range(0, len(result), 32)]
    return matrix
 
# 文件夹路径
SUT = sys.argv[1]
train_path = "/home/lddc/SHENYANLONG/live555/testProgs/output_"+SUT+"_RLGfuzz/train/"
seed_path = "/home/lddc/SHENYANLONG/live555/testProgs/output_"+SUT+"_RLGfuzz/queue/"

#清空之前的结果
if os.path.exists('Decode_Data/'+SUT+'/bitmap/'):
    clear_folder('Decode_Data/'+SUT+'/bitmap/')
else:
    os.mkdir('Decode_Data/'+SUT+'/bitmap/')

if os.path.exists('Decode_Data/'+SUT+'/statemap/'):
    clear_folder('Decode_Data/'+SUT+'/statemap/')
else:
    os.mkdir('Decode_Data/'+SUT+'/statemap/')

if os.path.exists('Decode_Data/'+SUT+'/seed/'):
    clear_folder('Decode_Data/'+SUT+'/seed/')
else:
    os.mkdir('Decode_Data/'+SUT+'/seed/')

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
            np.savetxt('Decode_Data/'+SUT+'/bitmap/'+file_name+'.txt', matrix, fmt='%d')
            #print(file_name+'已解析')
    if file_name.startswith("statemap"):
        # 构建文件的完整路径
        file_path = os.path.join(train_path, file_name)
        with open(file_path, "rb") as file:
            result = [0] * 256
            count = 0
            while True:
                data = file.read(4)
                if not data:
                    break
                value = int.from_bytes(data, byteorder='little')  # 将4字节数据转换为整数
                if value >0:
                    result[count] = 1
                else:
                    result[count] = 0

                count = count + 1
            result = np.array(result).reshape((16,16))
            result[0,0] = 1
            np.savetxt('Decode_Data/'+SUT+'/statemap/'+file_name+'.txt', result, fmt='%d')
            #print(file_name+'已解析')
print('------Statemaps and Bitmaps have been Decoded------')
for file_name in seed_names:
    if file_name.startswith("id"):
        file_path = os.path.join(seed_path, file_name)
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            decoded_text = codecs.decode(binary_data, 'ascii', 'ignore')

        with open('Decode_Data/'+SUT+'/seed/'+file_name+'.txt', 'w', encoding='ascii') as file:
            file.write(decoded_text)
        #print(file_name+'已解析')
print('------Seed Files have been Decoded------')


