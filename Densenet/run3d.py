import os
import torch
from data3D import *
from model3D import *
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.io import savemat

xulie = 31

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def getLevels(min, fac, num):
    return np.array([min*(fac**i) for i in range(num)])


def net_test_2d(net=None, device=None):
    device = device
    test_data = MyDataset(test_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)
    net.eval()


    for step, (src_data, trg_data,time_data_real,time_data_imag) in enumerate(test_loader):
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = src_data.to(device)
        trg_data = trg_data.to(device)

        time_data_real = time_data_real.type(torch.FloatTensor)
        time_data_real = time_data_real.to(device)

        time_data_imag = time_data_imag.type(torch.FloatTensor)
        time_data_imag = time_data_imag.to(device)
        [x_output] = net(src_data, time_data_real, time_data_imag)
        x_output = x_output.squeeze().detach().cpu().numpy()
        output = x_output

    out = output
    recon = torch.tensor(out)

    fig = plt.figure(figsize=(25, 10))
    contour_levels = 70

    ax = fig.add_subplot(121)
    contour_input = ax.contour(recon, levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of recon')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(122)
    contour_input = ax.contour(abs(ideal.squeeze()), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

    recon_spec = recon
    label_spec = abs(ideal.squeeze())

    recon_name = f'recon_spec{xulie}'
    file_path_recon = f'./A3DK08-8p-ft3_slices_recon/{recon_name}.mat'
    savemat(file_path_recon, {'recon_spec': recon_spec})
    label_name = f'label_spec{xulie}'
    file_path_label = f'./A3DK08-8p-ft3_slices_recon/{label_name}.mat'
    savemat(file_path_label, {'label_spec': label_spec})


if __name__ == "__main__":
    device = torch.device("cuda:4")
    N1 = 128
    array = 10
    total_output = torch.zeros(1, N2, N1)


    under = abs(torch.tensor(under))

    net = torch.load(r'dense_rate0.08_64_maskrandom_sumloss.pth')
    net = net.to(device)
    time_start = time.time()
    net_test_2d(net, device)
    time_end = time.time()
    time_cost = time_end - time_start
    print(time_cost)