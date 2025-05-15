import torch.nn as nn
from functools import partial
import torch
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = partial(nn.functional.interpolate,scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[-2] - x1.size()[-2]
        diffY = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class dual_path_unet(nn.Module):
    def __init__(self, in_chan = 1, out_chan = 1, chn = 64):
        super(dual_path_unet, self).__init__()
        self.inc = inconv(in_chan, chn)
        self.down1 = down(chn, 2*chn)
        self.down2 = down(2*chn, 4*chn)
        self.down3 = down(4*chn, 4*chn)

        self.inc_im = inconv(in_chan, chn)
        self.down1_im = down(chn, 2*chn)
        self.down2_im = down(2*chn, 4*chn)
        self.down3_im = down(4*chn, 4*chn)

        self.up1 = up(12*chn, 2*chn)
        self.up2 = up(4*chn, chn)
        self.up3 = up(2*chn, chn)
        self.outc = outconv(chn, out_chan)
    def forward(self, x, im):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        im1 = self.inc_im(im)
        im2 = self.down1_im(im1)
        im3 = self.down2_im(im2)
        im4 = self.down3_im(im3)

        x4 = torch.cat((x4,im4),dim=1)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
