import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Input(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Input, self).__init__()
        self.conv1 = SeparableConv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=2)  # 1→12
        self.bn1 = nn.BatchNorm2d(12)  # 12
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        return out


class upper_EDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upper_EDBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)  # 12
        self.conv1sep = SeparableConv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)  # 12→12
        self.relu1 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)  # 12
        self.conv2trans = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)  # 12→12
        self.relu2 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(in_channels * 2)  # 12 * 2 = 24
        self.conv3sep = SeparableConv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)  # 24→12
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        x1 = x
        out1 = self.bn1(x)
        out2 = self.conv1sep(out1)
        out3 = self.relu1(out2)
        out4 = self.bn2(out3)
        out5 = self.conv2trans(out4)
        out6 = self.relu2(out5)
        out7 = torch.cat((x1, out6), dim=1)  # 12 + 12 = 24
        out8 = self.bn3(out7)
        out9 = self.conv3sep(out8)
        out10 = self.relu3(out9)
        return out10


class down_EDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_EDBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)  # 12
        self.conv1sep = SeparableConv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)  # 12→12
        self.relu1 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(in_channels)  # 12
        self.conv2trans = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)  # 12→12
        self.relu2 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(in_channels * 2)  # 12 * 2 = 24
        self.conv3sep = SeparableConv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)  # 24→12
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        x1 = x
        out1 = self.bn1(x)
        out2 = self.conv1sep(out1)
        out3 = self.relu1(out2)
        out4 = self.bn2(out3)
        out5 = self.conv2trans(out4)
        out6 = self.relu2(out5)
        out7 = torch.cat((x1, out6), dim=1)  # 12 + 12 = 24
        out8 = self.bn3(out7)
        out9 = self.conv3sep(out8)
        out10 = self.relu3(out9)
        return out10


class hight_to_lowblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hight_to_lowblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)  # 12 or 24
        self.downconv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)  # e.g. 12→24 or 24→48
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.bn1(x)
        out = self.downconv(out)
        out = self.relu(out)
        return out


class low_to_hightblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(low_to_hightblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)  # 24 or 48 → 改为 24 or 12
        self.downconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)  # e.g. 24→12 or 48→24
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.bn1(x)
        out = self.downconv(out)
        out = self.relu(out)
        return out


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)  # 12→1

    def forward(self, x):
        out = self.conv(x)
        return out


class EDHRnet(nn.Module):
    def __init__(self):
        super(EDHRnet, self).__init__()
        self.input = Input(1, 12)  # 1 → 12

        self.edblock1upper = upper_EDBlock(12, 12)
        self.h_to_l1 = hight_to_lowblock(12, 24)  # 12 → 24

        self.edblock2upper = upper_EDBlock(12, 12)
        self.edblock3upper = upper_EDBlock(12, 12)

        self.h_to_l3 = hight_to_lowblock(12, 24)  # 12 → 24
        self.l_to_h3 = low_to_hightblock(24, 12)  # 24 → 12

        self.edblock4upper = upper_EDBlock(12, 12)
        self.edblock5upper = upper_EDBlock(12, 12)

        self.h_to_l5 = hight_to_lowblock(12, 24)  # 12 → 24
        self.l_to_h5 = low_to_hightblock(24, 12)  # 24 → 12

        self.edblock6upper = upper_EDBlock(12, 12)
        self.edblock7upper = upper_EDBlock(12, 12)
        self.edblock8upper = upper_EDBlock(12, 12)

        # down_EDBlock 输入是 24，输出也是 24
        self.edblock2down = down_EDBlock(24, 24)
        self.edblock3down = down_EDBlock(24, 24)
        self.edblock4down = down_EDBlock(24, 24)
        self.edblock5down = down_EDBlock(24, 24)
        self.edblock6down = down_EDBlock(24, 24)
        self.edblock7down = down_EDBlock(24, 24)

        self.l_to_h7 = low_to_hightblock(24, 12)  # 24 → 12

        self.out = Output(12, 1)  # 12 → 1

    def forward(self, x):
        x1 = self.input(x)
        x1_hight = self.edblock1upper(x1)
        x1_d = self.h_to_l1(x1_hight)

        x2_hight = self.edblock2upper(x1_hight)
        x3_hight = self.edblock3upper(x2_hight)

        x2_low = self.edblock2down(x1_d)
        x3_low = self.edblock3down(x2_low)

        x3_u = self.l_to_h3(x3_low)
        x3_d = self.h_to_l3(x3_hight)
        x3_hight1 = x3_hight + x3_u
        x3_low1 = x3_low + x3_d

        x4_hight = self.edblock4upper(x3_hight1)
        x5_hight = self.edblock5upper(x4_hight)
        x5_d = self.h_to_l5(x5_hight)
        x4_low = self.edblock4down(x3_low1)
        x5_low = self.edblock5down(x4_low)
        x5_u = self.l_to_h5(x5_low)
        x5_hight1 = x5_hight + x5_u
        x5_low1 = x5_low + x5_d

        x6_hight = self.edblock6upper(x5_hight1)
        x7_hight = self.edblock7upper(x6_hight)
        x6_low = self.edblock6down(x5_low1)
        x7_low = self.edblock7down(x6_low)
        x7_u = self.l_to_h7(x7_low)
        x7_hight1 = x7_hight + x7_u
        x8_hight = self.edblock8upper(x7_hight1)
        out = self.out(x8_hight)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)