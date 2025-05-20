import os
import sys
import time

cuda_id = 0  # Change to your cuda id.
num_workers = 0
exp_ids = [i for i in range(0, 10)]  # Change to your random seeds or other parameter names for data partitioning.

c = 1
sum = 0
total = int(len(exp_ids))
# os.chdir('../..')
os.makedirs('tmp', exist_ok=True)
for exp_id in exp_ids:
    print('\nProgress {}/{}'.format(c, total))
    start = time.time()
    # Be sure ArgumentParser is used for your work and change them.
    os.system(f'python main_ADCMT.py --exp_id {exp_id} >>tmp/tmp_ADCMT_YT-UGC_{cuda_id}.txt')
    # os.system(f'python main-KD.py --seed {exp_id} >tmp.txt')
    end = time.time()
    times = round((end-start)/60)
    print('It takes {} mins'.format(times))
    sum += times
    if c < total:  # 判断当前任务是否是最后一个
        ava = sum / c  # 平均每次耗时
        rest = round(ava * (total - c))  # 估算剩余时间
        print('Estimated remaining time: {} mins.'.format(rest))
    c += 1  # 累计任务计数
sys.exit()