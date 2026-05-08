import os
import torch
from data3D import MyDataset
from WNN3D import *
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
import scipy.io as scio

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def net_test_2d(net=None, device=None, current_xulie=None):
    device = device
    test_data = MyDataset(current_xulie)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)
    net.eval()


    path = os.path.join("Yfgj_i_slices", f"label_spec{current_xulie}.mat")


    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    matdata = scio.loadmat(path)
    global ideal
    ideal = matdata['spec']

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
    ideal = trg_data.squeeze().cpu().numpy()

    fig = plt.figure(figsize=(25, 10))
    contour_levels = 70

    ax = fig.add_subplot(121)
    contour_input = ax.contour(abs(recon), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of recon')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(122)
    contour_input = ax.contour(abs(ideal), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

    recon_spec = recon
    label_spec = abs(ideal.squeeze())

    recon_name = f'recon_spec{xulie}'
    file_path_recon = f'./Yfgj_i_slices_recon/{recon_name}.mat'
    savemat(file_path_recon, {'recon_spec': recon_spec})
    label_name = f'label_spec{xulie}'
    file_path_label = f'./Yfgj_i_slices_recon/{label_name}.mat'
    savemat(file_path_label, {'label_spec': label_spec})
    return time.time()


if __name__ == "__main__":
    device = torch.device("cuda:3")
    N1 = 64

    # 加载模型
    print("Loading model...")
    net = torch.load(r'WNN_0.08_34_34_maskrandom.pth')
    net = net.to(device)

    successful_runs = 0
    total_time = 0

    # 确保输出目录存在
    os.makedirs('./Yfgj_i_slices_recon', exist_ok=True)

    peak_position = [382, 388, 393, 394, 397, 399, 402, 403, 406, 408,
                     411, 412, 414, 415, 416, 417, 420, 421, 424, 428,
                     429, 430, 432, 433, 435, 437, 439, 441, 442, 443,
                     444, 445, 447, 452, 453, 456, 458, 462]
    # for xulie in peak_position:
    for xulie in range(1, 522):
        try:
            time_start = time.time()
            end_time = net_test_2d(net, device, xulie)
            time_cost = end_time - time_start
            total_time += time_cost
            successful_runs += 1

            print(f"xulie {xulie} completed in {time_cost:.2f} seconds")

        except Exception as e:
            print(f"Error processing xulie {xulie}: {str(e)}")
            continue

    # 输出统计信息
    print(f"\nBatch processing completed!")
    print(f"Total time: {total_time:.2f} seconds")
    if successful_runs > 0:
        print(f"Average time per case: {total_time / successful_runs:.2f} seconds")