import math
import random
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'

class MyDataset(dataset.Dataset):
    def __init__(self, xulie, data_dir="A3DK08_r_slices"):
        self.xulie = xulie
        self.data_dir = data_dir

        # 加载数据
        path = os.path.join(data_dir, f"label_spec{xulie}.mat")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        matdata = scio.loadmat(path)
        ideal = matdata['spec']


        N1 = 64
        pad_zeros = N1 - ideal.shape[1]
        N2 = ideal.shape[0]

        # 采样模板
        result_formatted = random.randint(1, 1000)
        mask_path = f'./0.08_2D_mask-40-64/Mask_{result_formatted}.mat'
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        Mask = scio.loadmat(mask_path)
        U = Mask['Mask']
        under = np.multiply(np.fft.ifft2(ideal), U)
        xx_real = np.real(under)
        xx_imag = np.imag(under)

        under = np.fft.fft2(under)

        max_amp = np.max(np.abs(under))
        under = np.abs(under) / (max_amp)
        ideal = np.abs(ideal) / (max_amp *12.5)
        xx_real = xx_real / (max_amp *12.5)
        xx_imag = xx_imag / (max_amp *12.5)

        under = np.expand_dims(under, axis=0)
        ideal = np.expand_dims(ideal, axis=0)
        xx_real = np.expand_dims(xx_real, axis=0)
        xx_imag = np.expand_dims(xx_imag, axis=0)

        self.test_list = np.stack([under, ideal, xx_real, xx_imag], axis=0)
        self.data_lengths = 1

    def __getitem__(self, index):
        data = self.test_list
        src_data = data[0]
        trg_data = data[1]
        time_data_real = data[2]
        time_data_imag = data[3]
        return src_data, trg_data, time_data_real, time_data_imag

    def __len__(self):
        return self.data_lengths

xulie_global = 1
test_data_global = MyDataset(xulie_global)