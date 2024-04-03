import os
import shutil
import time

def copy_and_rename_file(source_file, destination_folder):
    # 获取当前时间
    current_time = time.strftime('%Y%m%d_%H%M%S')
    # 构建目标文件名
    destination_file = os.path.join(destination_folder, f'{current_time}.txt')
    # 复制文件
    shutil.copy2(source_file, destination_file)
    print(f'Copied and renamed {source_file} to {destination_file}')

def main():
    # 源文件路径
    source_file = '/home/lddc/SHENYANLONG/live555/testProgs/output/fuzzer_stats'
    # 目标文件夹路径
    current_time = time.strftime('%Y%m%d_%H')
    destination_folder = 'Train_Result/fuzz_stats_'+current_time
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    while True:
        # 复制并重命名文件
        copy_and_rename_file(source_file, destination_folder)
        time.sleep(600)  # 10分钟保存一次

if __name__ == "__main__":
    main()
