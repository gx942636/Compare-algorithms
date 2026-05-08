import torch
import numpy as np
import scipy.io as scio
import random
import scipy
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
import h5py
from matplotlib import pyplot as plt

class MyDataset(dataset.Dataset):
    def __init__(self, num_samples, split='train'):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        NOISE_input, GT_label, factor_matrix= self.__gen_signal__(idx)

        max_amp = np.max(np.abs(NOISE_input))
        NOISE_input = np.abs(NOISE_input) / (max_amp)
        GT_label = np.abs(GT_label) / (max_amp * 12.5)

        NOISE_input = np.expand_dims(NOISE_input, axis=0)
        GT_label = np.expand_dims(GT_label, axis=0)
        factor_matrix = np.expand_dims(factor_matrix, axis=0)

        return NOISE_input, GT_label, factor_matrix

    def __gen_signal__(self, idx):
        dim = 2

        # N = 64
        N1 = 40  # 第一个维度大小
        N2 = 64  # 第二个维度大小
        max_J = 40
        # max_J = random.randint(350, 550)

        # Generate FID signals
        J = np.random.randint(5, random.randint(7, 14), size=(dim, 1))  # Random number of harmonics
        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel() - 1] = 1  # TODO
        mask = np.cumsum(mask, axis=1)

        ph = np.random.uniform(0.0, 0.2 * np.pi, size=(dim, max_J))  # Random phase  # TODO
        A = np.random.uniform(0.02, 1.0, size=(dim, max_J))  # Random amplitude
        w = np.random.uniform(0.05, 0.95, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time

        t1 = np.arange(N1)  # Time axis
        t2 = np.arange(N2)  # Time axis

        A = np.multiply(A, mask)

        x1 = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t1 / sgm[..., None]) * np.exp(
            1j * 2 * np.pi * w[..., None] * t1)
        x2 = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t2 / sgm[..., None]) * np.exp(
            1j * 2 * np.pi * w[..., None] * t2)
        # xn_unit = np.matmul(x1[0][:, :, np.newaxis], x2[0][:, np.newaxis])
        xn_unit = np.matmul(x1[0][:, :, np.newaxis], x2[1][:, np.newaxis])
        clean_xn = np.sum(xn_unit, axis=0)

        # Add noise to FID signals
        noise_scale = 1e-4
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(N1, N2))
        xx = noise + clean_xn

        threshold = 3
        # Create binary masks for the two dimensions
        temp1 = np.zeros((dim, max_J, N1))
        temp2 = np.zeros((dim, max_J, N2))

        # 减少时间复杂度
        for i in range(dim):
            freq_indices1 = (np.abs(2 * np.pi * w[i, :] - (np.linspace(0, 1, N1) * 2 * np.pi)[:, None])
                             <= threshold / sgm[i, :])
            freq_indices1 = freq_indices1.T
            temp1[i, freq_indices1] = 1
            freq_indices2 = np.abs(
                2 * np.pi * w[i, :] - (np.linspace(0, 1, N2) * 2 * np.pi)[:, None]) <= threshold / sgm[i, :]
            freq_indices2 = freq_indices2.T
            temp2[i, freq_indices2] = 1

        # Combine binary peak presence masks across dimensions
        A1 = np.multiply(np.ones((dim, max_J)), mask)
        temp_l1l2 = np.matmul(A1[0][:, np.newaxis, np.newaxis] * temp1[0][:, :, np.newaxis],
                              A1[1][:, np.newaxis, np.newaxis] * temp2[1][:, np.newaxis])
        factor_matrix = np.sum(temp_l1l2, axis=0)
        factor_matrix[factor_matrix > 1] = 1

        # 采样率固定
        # result_formatted = random.randint(1, 1000)  # 生成1到10000之间的随机整数
        result_formatted = 50
        Mask = scio.loadmat('./0.08_2D_mask-40-64/Mask_' + str(result_formatted) + '.mat')['Mask']
        U = Mask

        NUS_FID = np.multiply(U, xx)
        # NUS_FID_pad = np.pad(NUS_FID, ((0, (N - N1)), (0, (N - N2))), mode='constant', constant_values=0)

        NOISE_FID = np.multiply(U, xx)
        # NOISE_FID = np.pad(NOISE_FID, ((0, (N - N1)), (0, (N - N2))), mode='constant', constant_values=0)
        NOISE_input = np.fft.fft2(NOISE_FID)
        GT_label = np.fft.fft2(xx)
        return NOISE_input, GT_label, factor_matrix


def create_dataset(phase):
    '''create dataset'''

    if phase == 'train':
        dataset = MyDataset(
                    num_samples=40000,
                    split=phase,
                                    )
    elif phase == 'val':
        dataset = MyDataset(num_samples=5000,
                    split=phase,)
    return dataset

def create_dataloader(dataset, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False)

