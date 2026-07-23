import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, attention=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear, attention)
        self.up2 = Up(512, 256 // factor, bilinear, attention)
        self.up3 = Up(256, 128 // factor, bilinear, attention)
        self.up4 = Up(128, 64, bilinear, attention)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """（卷积 => [BN] => ReLU）重复 2 次"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """先最大池化，再进行双卷积的下采样模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """注意力门（Attention U-Net）：用解码器特征 g 作为门控信号，
    对跳跃连接特征 x 逐像素生成 0~1 权重，抑制背景区域、保留目标区域。"""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # 两路特征各自降维到 inter_channels 后相加，再压成单通道注意力图
        attention = self.psi(self.relu(self.w_gate(gate) + self.w_skip(skip)))
        return skip * attention


class Up(nn.Module):
    """先上采样，再与跳跃连接特征拼接并做双卷积；可选在拼接前对跳跃特征过注意力门"""

    def __init__(self, in_channels, out_channels, bilinear=True, attention=False):
        super().__init__()

        # 如果使用双线性上采样，则用普通卷积来减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # 上采样后的门控特征与跳跃特征通道数均为 in_channels // 2
        self.attention_gate = (
            AttentionGate(in_channels // 2, in_channels // 2, in_channels // 4)
            if attention else None
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入张量格式为 CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.attention_gate is not None:
            x2 = self.attention_gate(gate=x1, skip=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """解码块：双线性上采样 → 可选注意力门 → 与跳跃特征拼接 → 双卷积。"""

    def __init__(self, in_channels, skip_channels, out_channels, attention=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention_gate = (
            AttentionGate(in_channels, skip_channels, max(skip_channels // 2, 1))
            if attention else None
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 上采样后与跳跃特征尺寸可能相差 1 像素，补边对齐
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        if self.attention_gate is not None:
            skip = self.attention_gate(gate=x, skip=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """ResNet-34（ImageNet 预训练）编码器 + U-Net 解码器，可选注意力门。

    编码器复用 torchvision 的 resnet34，五级特征分别为
    stem 64/2、layer1 64/4、layer2 128/8、layer3 256/16、layer4 512/32；
    解码器逐级上采样并与对应跳跃特征融合，最后再上采样一次恢复到输入分辨率。
    """

    def __init__(self, n_channels=3, n_classes=1, attention=False, pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.attention = attention

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        if n_channels != 3:
            backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # ---------- 编码器（取自 resnet34） ----------
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64,  /2
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64,  /4
        self.layer2 = backbone.layer2  # 128, /8
        self.layer3 = backbone.layer3  # 256, /16
        self.layer4 = backbone.layer4  # 512, /32

        # ---------- 解码器 ----------
        self.dec4 = DecoderBlock(512, 256, 256, attention)  # /16
        self.dec3 = DecoderBlock(256, 128, 128, attention)  # /8
        self.dec2 = DecoderBlock(128, 64, 64, attention)    # /4
        self.dec1 = DecoderBlock(64, 64, 32, attention)     # /2
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # /1
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        e0 = self.stem(x)                 # 64,  /2
        e1 = self.layer1(self.pool(e0))   # 64,  /4
        e2 = self.layer2(e1)              # 128, /8
        e3 = self.layer3(e2)              # 256, /16
        e4 = self.layer4(e3)              # 512, /32
        d4 = self.dec4(e4, e3)            # 256, /16
        d3 = self.dec3(d4, e2)            # 128, /8
        d2 = self.dec2(d3, e1)            # 64,  /4
        d1 = self.dec1(d2, e0)            # 32,  /2
        d0 = self.up0(d1)                 # 32,  /1
        return self.outc(d0)
