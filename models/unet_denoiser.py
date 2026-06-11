import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionResidualUNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64,128)

        self.bottleneck = DoubleConv(128,256)

        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.dec1 = DoubleConv(256,128)

        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = DoubleConv(128,64)

        self.final = nn.Conv2d(64,out_channels,1)

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1,e2],dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2,e1],dim=1)
        d2 = self.dec2(d2)

        return self.final(d2)