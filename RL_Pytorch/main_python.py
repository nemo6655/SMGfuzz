import time
for t in range(0,50):
    time.sleep(60)
    print('Waiting for running SMGfuzz, has been waiting for '+str(t)+'mins')

for i in range(0,48):
    time_start = time.time()
    exec(open("read_binary_file.py").read())
    exec(open("Test_word2vec.py").read())
    exec(open("train.py").read())
    for j in range(0,5):
        exec(open("d3qn.py").read())
    time_end = time.time()
    time_delta = time_end-time_start
    if time_delta < 1800:
        time.sleep(1800-time_delta)
        
    