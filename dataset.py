from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from torch.utils.data import Dataset, DataLoader
import math
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(0)
style.use('ggplot')


class ToyDataset(Dataset):
    def __init__(self, data_num):
        self.data_num = data_num
        class_0_data_major = np.random.normal((0.30, 0.5), (0.1, 0.08), size=(int(data_num * 3 / 16), 2)).reshape((-1, 2))
        class_0_data_minor = np.random.normal((0.5, 0.65), (0.05, 0.05), size=(int(data_num / 16), 2)).reshape((-1, 2))
        self.data_0 = np.concatenate([class_0_data_major, class_0_data_minor], axis=0)
        class_1_data_major = np.random.normal((0.70, 0.5), (0.1, 0.08), size=(int(data_num * 3 / 16), 2)).reshape(-1, 2)
        class_1_data_minor = np.random.normal((0.5, 0.35), (0.05, 0.05), size=(int(data_num / 16), 2)).reshape(-1, 2)
        self.data_1 = np.concatenate([class_1_data_major, class_1_data_minor], axis=0)
        class_2_data_fhalf = self.select_out_data(0.5, 0.35, 0.06, 0.1, 0.12, data_num / 4).reshape(-1, 2)
        class_2_data_shalf = self.select_out_data(0.5, 0.65, 0.06, 0.1, 0.12, data_num / 4).reshape(-1, 2)
        self.data_2 = np.concatenate([class_2_data_fhalf, class_2_data_shalf], axis=0)
        self.data = np.concatenate([self.data_0, self.data_1, self.data_2], axis=0)
        label_0 = np.zeros((self.data_0.shape[0], 1))
        label_1 = np.ones((self.data_1.shape[0], 1))
        label_2 = np.ones((self.data_2.shape[0], 1)) * 2
        print(label_0.shape)
        print(label_1.shape)
        print(label_2.shape)
        self.label = np.concatenate([label_0, label_1, label_2], axis=0)
        assert self.data.shape[0] == self.label.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def select_out_data(self, target_mean_x, target_mean_y, target_std_x, target_std_y, r, total_num):
        data_array = []
        while True:
            x = np.random.normal((target_mean_x, target_mean_y), (target_std_x, target_std_y), size=(2, ))
            #print(x)
            dist = math.sqrt((x[0] - target_mean_x) ** 2 + (x[1] - target_mean_y) ** 2)
            if dist > r:
                data_array.append(x)
            if len(data_array) >= total_num:
                break
        data_array = np.array(data_array)
        return data_array

    def visualize_data(self):
        # fig = plt.figure()
        plt.scatter(self.data_0[:, 0], self.data_0[:, 1], color='red', s=1.5)
        plt.scatter(self.data_1[:, 0], self.data_1[:, 1], color='yellow', s=1.5)
        plt.scatter(self.data_2[:, 0], self.data_2[:, 1], color='blue', s=1.5)
        plt.show()
        # fig.savefig('./figure/toy_data.png')

# train_data = ToyDataset(960)
# print(train_data.data.shape)
# train_data.visualize_data()