import numpy as np
import glob
import os
import h5py
import torch
import time
import spectral_attack
import warnings
warnings.filterwarnings("ignore")

stage = {'test', 'train'}


def GFT2(point):
    K = 10
    v, laplacian, u = spectral_attack.eig_vector(point, K)  # v 特征向量组成的正交矩阵 ， u 特征值
    return v, laplacian, u

for partition in stage:
    k = 0

    path = "Created_Data\\{}_data\\".format(partition)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for h5_name in glob.glob(os.path.join('data\\modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')[:, 0:1024, :]
        label = f['label'][:].astype('int64')
        f.close()
        all_data2 = []
        v_list = []
        laplacian_list = []
        u_list = []
        time_total = 0
        count = 0

        for i in range(len(data)):
            count += 1
            start_time = time.time()
            v, laplacian, u = GFT2(torch.tensor([data[i]]).to("cuda"))
            end_time = time.time()
            used_time = end_time - start_time
            time_total += used_time

            v_list.append(v[0].cpu().numpy())
            laplacian_list.append(laplacian[0].cpu().numpy())
            u_list.append(u[0].cpu().numpy())

            dic = {'data': data[i], 'label': label[i], 'v': v[0].cpu().numpy()}
            dic = np.array(list(dic.items()))   # pip install numpy==1.23.5

            np.save(path + '{}{}'.format(partition,k), dic)
            k += 1

            print(k)




