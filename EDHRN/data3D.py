import math
import random
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, xulie, data_dir="Yfgj_i_slices"):
        self.xulie = xulie
        self.data_dir = data_dir

        # 加载数据
        path = os.path.join(data_dir, f"label_spec{xulie}.mat")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        matdata = scio.loadmat(path)
        ideal = matdata['spec']
        ideal1 = ideal.copy()

        N1 = 64
        pad_zeros = N1 - ideal.shape[1]
        N2 = ideal.shape[0]

        # 采样模板
        # result_formatted = random.randint(1, 1000)  # 生成1到10000之间的随机整数
        result_formatted = 50  # 可以改为根据xulie变化
        mask_path = f'./0.08_2D_mask-34-34/Mask_{result_formatted}.mat'

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        Mask = scio.loadmat(mask_path)
        U = Mask['Mask'].astype(np.float32)

        under = np.multiply(np.fft.ifft2(ideal), U)
        under = np.abs(np.fft.fft2(under))

        # 添加高斯噪声
        sigma = 15 * np.max(under)
        noise = np.random.normal(0, sigma, under.shape).astype(np.float32)
        under_noisy = under + noise

        max_amp = np.max(np.abs(under))
        under = np.abs(under) / (max_amp)
        ideal = np.abs(ideal) / (max_amp * 12.5)

        # 调整维度
        under = np.expand_dims(under, axis=0)
        ideal = np.expand_dims(ideal, axis=0)

        self.test_list = np.stack([under, ideal], axis=0)  # 形状变为 [2, 1, H, W]
        self.data_lengths = 1

    def __getitem__(self, index):
        data = self.test_list
        src_data = data[0]
        trg_data = data[1]
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths


def create_dataset(xulie=None):
    if xulie is None:
        import sys
        if hasattr(sys.modules['__main__'], 'xulie'):
            xulie = sys.modules['__main__'].xulie
        else:
            xulie = 1

    return MyDataset(xulie)

xulie_global = 1
test_data_global = MyDataset(xulie_global)