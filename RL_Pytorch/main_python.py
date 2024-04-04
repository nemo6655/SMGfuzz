import time
import argparse
import sys
parser = argparse.ArgumentParser(description='input SUT name')
# 添加文件夹路径参数
parser.add_argument('--SUT', type=str, help='input SUT name')
# 解析命令行参数
args = parser.parse_args()
print('------------ The SUT is '+ args.SUT+'------------')
for t in range(0,110):
    print('------Waiting for running SMGfuzz, has been waiting for '+str(t+1)+' mins------')
    #time.sleep(60)

for i in range(0,48):
    time_start = time.time()
    sys.argv = ["read_binary_file.py", args.SUT]
    exec(open("read_binary_file.py").read())
    
    sys.argv = ["Test_word2vec.py", args.SUT]
    exec(open("Test_word2vec.py").read())
    
    sys.argv = ["train.py", args.SUT]
    exec(open("train.py").read())
    
    sys.argv = ["d3qn.py", args.SUT]
    for j in range(0,2):
        exec(open("d3qn.py").read())
    time_end = time.time()
    time_delta = time_end-time_start
    if time_delta < 1800:
        print('------Python thread will standby '+str(int(1800-time_delta))+' seconds ------')
        #time.sleep(1750-time_delta)
        
        
    