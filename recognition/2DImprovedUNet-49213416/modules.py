import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_ch, out_ch, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )

class ContextBlock(nn.Module):
 
    def __init__(self, channels, dropout=0.3):
        super().__init__
        self.block = nn.Sequential(conv(channels,channels), nn.Dropout2d(dropout), conv(channels,channels))
 
    def forward(self, x):
        return x + self.block(x)

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = conv(in_ch, out_ch)
        self.localize = nn.Sequential(conv(out_ch * 2, out_ch * 2), conv(out_ch * 2, out_ch, kernel_size=1))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")  
        x = self.up(x)
        return self.localize(torch.cat([skip, x], dim=1))


class ImprovedUNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        self.enc1 = nn.Sequential(conv(in_channels, 16), ContextBlock(16))
        self.enc2 = nn.Sequential(conv(16, 32, stride=2), ContextBlock(32))
        self.enc3 = nn.Sequential(conv(32, 64, stride=2), ContextBlock(64))
        self.enc4 = nn.Sequential(conv(64, 128, stride=2), ContextBlock(128))
        self.bottom = nn.Sequential(conv(128, 256, stride=2), ContextBlock(256))

        self.dec4 = UpSample(256, 128)
        self.dec3 = UpSample(128, 64)
        self.dec2 = UpSample(64, 32)
        self.dec1 = UpSample(32, 16)

        self.head3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head1 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottom(e4)
 
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = F.interpolate(self.head3(d3), size=size, mode="nearest")
        out = out + F.interpolate(self.head2(d2), size=size, mode="nearest")
        out = out + self.head1(d1)
        return out