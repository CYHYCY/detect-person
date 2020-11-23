import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        """
        x.tanh(ln(1+exp(x)))
        :param x:
        :return:
        """
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """
        conv+bn+activation
        :param in_channels: 输入的通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        """
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock(nn.Module):
    """
    高宽不变，通道数不变
    """

    def __init__(self, channels, hidden_channels=None, residual_activation=nn.Identity()):
        """

        :param channels:
        :param hidden_channels:
        :param residual_activation:
        """
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),  # 高宽不变
            BasicConv(hidden_channels, channels, 3)  # 高宽不变
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param num_blocks: 多少个block
        :param first: 标志位
        """
        super(Resblock_body, self).__init__()
        # (N, out_channels, ceil(H/2), ceil(W/2))
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)  # 高宽不变，通道数为out_channels
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  # 高宽不变，通道数为out_channels
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2),  # 高宽不变，通道数不变
                BasicConv(out_channels, out_channels, 1)  # 高宽不变，通道数不变
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)  # 高宽不变、通道数为out_channels
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)  # 高宽不变、通道数为floor(out_channels/2)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)  # 高宽不变、通道数为floor(out_channels/2)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels // 2) for _ in range(num_blocks)],  # 高宽不变、通道数为floor(out_channels/2)
                BasicConv(out_channels // 2, out_channels // 2, 1)  # 高宽不变、通道数为floor(out_channels/2)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)  # 高宽不变、通道数为out_channels

    def forward(self, x):
        """
        最终输出feature map shape is (N, out_channels, ceil(H/2), ceil(W/2))
        :param x: shape is (N,C,H,W)
        :return: x: shape is (N, out_channels, ceil(H/2), ceil(W/2))
        """
        x = self.downsample_conv(x)
        # (N, out_channels, ceil(H/2), ceil(W/2)) || (N, out_channels//2, ceil(H/2), ceil(W/2))

        x0 = self.split_conv0(x)
        # (N, out_channels, ceil(H/2), ceil(W/2))  ||  (N, out_channels//2, ceil(H/2), ceil(W/2))

        x1 = self.split_conv1(x)
        # (N, out_channels, ceil(H/2), ceil(W/2))  ||  (N, out_channels//2, ceil(H/2), ceil(W/2))
        x1 = self.blocks_conv(x1)
        # (N, out_channels, ceil(H/2), ceil(W/2))  ||  (N, out_channels//2, ceil(H/2), ceil(W/2))

        x = torch.cat([x1, x0], dim=1)
        # (N, 2*out_channels, ceil(H/2), ceil(W/2))  ||  (N, out_channels, ceil(H/2), ceil(W/2))
        x = self.concat_conv(x)  # (N, out_channels, ceil(H/2), ceil(W/2))

        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)  # 高宽不变，通道数变为inplanes
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():  # 采用深度优先遍历的方式，存储了net的所有模块
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)  # 高宽不变，通道数变为inplanes

        x = self.stages[0](x)  # 高宽减半，通道数变为feature_channels[0]
        x = self.stages[1](x)  # 高宽减半，通道数变为feature_channels[1]
        out3 = self.stages[2](x)  # 高宽减半，通道数变为feature_channels[2]
        out4 = self.stages[3](out3)  # 高宽减半，通道数变为feature_channels[3]
        out5 = self.stages[4](out4)  # 高宽减半，通道数变为feature_channels[4]

        return out3, out4, out5


def darknet53(pretrained, **kwargs):
    """
    分别8倍降，16倍降，32倍降。最终feature map的高宽向上取整
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    ipt = torch.randn(1, 3, 608, 608)
    model = CSPDarkNet([1, 2, 8, 8, 4])
    out3, out4, out5 = model(ipt)
