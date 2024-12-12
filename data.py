#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import random
import math
import torch

import pyvista as pv

def show_pl(pointclouds_pl_adv, special_index):
    pointclouds_pl_adv=pointclouds_pl_adv.squeeze()
    p = pv.Plotter()
    camera = pv.Camera()
    camera.position = (18,4,-20)
    camera.focal_point = (0,0,0)
    for i in range(len(pointclouds_pl_adv)):
        p.add_mesh(pv.PolyData(pointclouds_pl_adv[i]), color=[0, 0, 0], point_size=np.float(11), render_points_as_spheres=True)
    p.camera = camera
    p.show()



def generate_point_with_distance(point, distance):
    x1, y1, z1 = point[0], point[1], point[2]

    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)

    x2 = x1 + distance * math.sin(phi) * math.cos(theta)
    y2 = y1 + distance * math.sin(phi) * math.sin(theta)
    z2 = z1 + distance * math.cos(phi)

    return np.array([x2, y2, z2])



def load_dir(partition, test_poison):
    DATA_DIR = 'Created_Data'
    DATA_DIR = DATA_DIR + "\\" + partition + "_data"
    paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            paths.append(file_path)

    if partition == 'test' and test_poison == True:
        paths = np.array(paths)
    return paths



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, test_poison, test_poison_target,target_num, model, GFT_noise, partition='train'):
        #### Read the index of the poisoned sample
        self.poison_train_index = np.load('modelnet40_poison_train_index.npy') # the index of poisoned training data. 5% poison rate
        self.test_poison = test_poison
        self.test_poison_target = test_poison_target
        self.model = model
        self.GFT_noise = GFT_noise
        self.poison_train_index_list = []
        self.center_num_list = []
        self.target_list = []
        self.paths = load_dir(partition, test_poison)
        self.num_points = num_points
        self.partition = partition
        self.target_label = 2



    def __getitem__(self, item):
        path = self.paths[item]
        Data = np.load(path, allow_pickle=True)
        pointcloud = Data[0][1]
        label = Data[1][1]
        label = torch.tensor(np.array(label))
        v = Data[2][1]
        conduct_poison = False
        if self.partition == 'train':
            if item in self.poison_train_index:
                conduct_poison = True
        pointcloud = np.float32(pointcloud)
        return pointcloud, label, v, conduct_poison
    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
