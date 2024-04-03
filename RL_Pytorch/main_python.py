import time
for t in range(0,115):
    print('------Waiting for running SMGfuzz, has been waiting for '+str(t+1)+' mins------')
    #time.sleep(60)

for i in range(0,48):
    time_start = time.time()
    exec(open("read_binary_file.py").read())
    exec(open("Test_word2vec.py").read())
    exec(open("train.py").read())
    for j in range(0,2):
        exec(open("d3qn.py").read())
    time_end = time.time()
    time_delta = time_end-time_start
    if time_delta < 1800:
        print('------Python thread will standby '+str(int(1800-time_delta))+' seconds ------')
        #time.sleep(1800-time_delta)
        if os.path.exists('Train_Result/RL_Result/'):
            clear_folder('Train_Result/RL_Result/')
        else:
            os.mkdir('Train_Result/RL_Result/')
        
    