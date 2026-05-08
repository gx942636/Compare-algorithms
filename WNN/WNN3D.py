import torch
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

class Aconv2d(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(Aconv2d, self).__init__()

        assert activate in ['sigmoid', 'tanh']

        self.dilation = dilation
        self.activate = activate

        self.dilation_conv2d = torch.nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=(3, 3),
            dilation=(dilation, dilation),
            bias=False
        )

    def forward(self, inputs):
        pad_h = self.dilation
        pad_w = self.dilation
        inputs = F.pad(inputs, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        outputs = self.dilation_conv2d(inputs)
        return outputs


class ResnetBlock(torch.nn.Module):
    def __init__(self, dilation, channel_in, channel_out):
        super(ResnetBlock, self).__init__()
        self.conv_relu = Aconv2d(dilation=dilation, channel_in=channel_in, channel_out=channel_out)

    def forward(self, x):
        output = self.conv_relu(x)
        outputs = F.relu(output)
        return outputs

class MyModuleList(torch.nn.Module):
    def __init__(self, dilations=[2, 4, 8, 16, 16]):
        super(MyModuleList, self).__init__()
        self.layer0 = ResnetBlock(dilation=1, channel_in=1, channel_out=20)
        self.layers = torch.nn.ModuleList([
            ResnetBlock(dilation=dilation, channel_in=20, channel_out=20) for dilation in dilations
        ])
        self.layer7 = ResnetBlock(dilation=16, channel_in=20, channel_out=20)
        self.convout = torch.nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.layer0(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer7(x)
        out = self.convout(x)
        return out

class WNN(torch.nn.Module):
    def __init__(self, num_classes, channels_in=1, channels_out=20, dilations=[1,2,4,8,16]):
        super(WNN, self).__init__()

        self.resnet_block_0 = MyModuleList(dilations=[2,4,8,16,16])
        self.resnet_block_1 = MyModuleList(dilations=[2,4,8,16,16])
        self.resnet_block_2 = MyModuleList(dilations=[2,4,8,16,16])

    def forward(self, inputs, time_data_real, time_data_imag):
        x0 = self.resnet_block_0(inputs)
        mask = (time_data_real != 0.0).squeeze(1)
        complex = torch.view_as_complex(torch.stack([time_data_real, time_data_imag], dim=-1)).squeeze(1)
        xxx = torch.fft.ifft2(x0, dim=(-2, -1), norm=None).squeeze(1)

        xxx[mask] = complex[mask]
        time_pred = xxx

        out0 = torch.real(torch.fft.fft2(time_pred, dim=(-2, -1), norm=None)).unsqueeze(1)

        x1 = self.resnet_block_1(out0)
        mask = (time_data_real != 0.0).squeeze(1)
        complex = torch.view_as_complex(torch.stack([time_data_real, time_data_imag], dim=-1)).squeeze(1)
        xxx = torch.fft.ifft2(x1, dim=(-2, -1), norm=None).squeeze(1)

        xxx[mask] = complex[mask]
        time_pred = xxx

        out1 = torch.real(torch.fft.fft2(time_pred, dim=(-2, -1), norm=None)).unsqueeze(1)

        x2 = self.resnet_block_2(out1)

        return x2