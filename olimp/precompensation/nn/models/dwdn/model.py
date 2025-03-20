import torch.nn as nn
import torch
import torch.nn.functional as F

from .basicblock import get_uperleft_denominator


class Conv(nn.Module):
    def __init__(
        self,
        input_channels,
        n_feats,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        bn=False,
        act=False,
    ):
        super(Conv, self).__init__()
        m = []
        m.append(
            nn.Conv2d(
                input_channels,
                n_feats,
                kernel_size,
                stride,
                padding,
                bias=bias,
            )
        )
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class Deconv(nn.Module):
    def __init__(
        self,
        input_channels,
        n_feats,
        kernel_size,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
        act=False,
    ):
        super(Deconv, self).__init__()
        m = []
        m.append(
            nn.ConvTranspose2d(
                input_channels,
                n_feats,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            )
        )
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        padding=0,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                conv(n_feat, n_feat, kernel_size, padding=padding, bias=bias)
            )
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class DWDN(nn.Module):
    def __init__(self, n_levels: int, scale: float):
        super().__init__()

        n_resblock = 3
        n_feats1 = 16
        n_feats = 32
        kernel_size = 5

        self.n_colors = 3
        self.n_levels = n_levels
        self.scale = scale

        FeatureBlock = [
            Conv(self.n_colors, n_feats1, kernel_size, padding=2, act=True),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
        ]

        InBlock1 = [
            Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
        ]
        InBlock2 = [
            Conv(
                n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True
            ),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
        ]

        # encoder1
        Encoder_first = [
            Conv(
                n_feats,
                n_feats * 2,
                kernel_size,
                padding=2,
                stride=2,
                act=True,
            ),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
        ]
        # encoder2
        Encoder_second = [
            Conv(
                n_feats * 2,
                n_feats * 4,
                kernel_size,
                padding=2,
                stride=2,
                act=True,
            ),
            ResBlock(Conv, n_feats * 4, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 4, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 4, kernel_size, padding=2),
        ]
        # decoder2
        Decoder_second = [
            ResBlock(Conv, n_feats * 4, kernel_size, padding=2)
            for _ in range(n_resblock)
        ]
        Decoder_second.append(
            Deconv(
                n_feats * 4,
                n_feats * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                act=True,
            )
        )

        # decoder1
        Decoder_first = [
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2)
            for _ in range(n_resblock)
        ]
        Decoder_first.append(
            Deconv(
                n_feats * 2,
                n_feats,
                kernel_size=3,
                padding=1,
                output_padding=1,
                act=True,
            )
        )

        OutBlock = [
            ResBlock(Conv, n_feats, kernel_size, padding=2)
            for _ in range(n_resblock)
        ]
        OutBlock2 = [Conv(n_feats, self.n_colors, kernel_size, padding=2)]

        self.FeatureBlock = nn.Sequential(*FeatureBlock)
        self.inBlock1 = nn.Sequential(*InBlock1)
        self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock2 = nn.Sequential(*OutBlock2)

    def forward(self, input, kernel):
        for jj in range(kernel.shape[0]):
            kernel[jj : jj + 1, :, :, :] = torch.div(
                kernel[jj : jj + 1, :, :, :],
                torch.sum(kernel[jj : jj + 1, :, :, :]),
            )
        feature_out = self.FeatureBlock(input)
        clear_features = torch.zeros(feature_out.size())
        ks = kernel.shape[2]
        dim = (ks, ks, ks, ks)
        first_scale_inblock_pad = F.pad(feature_out, dim, "replicate")
        for i in range(first_scale_inblock_pad.shape[1]):
            blur_feature_ch = first_scale_inblock_pad[:, i : i + 1, :, :]
            clear_feature_ch = get_uperleft_denominator(
                blur_feature_ch, kernel
            )
            clear_features[:, i : i + 1, :, :] = clear_feature_ch.real[
                :, :, ks:-ks, ks:-ks
            ]

        output = []
        for level in range(self.n_levels):
            scale = self.scale ** (self.n_levels - level - 1)
            _, _, h, w = input.shape
            hi = round(h * scale)
            wi = round(w * scale)

            if level == 0:
                input_clear = F.interpolate(
                    clear_features, (hi, wi), mode="bilinear"
                )
                inp_all = input_clear.to(input.device)
                first_scale_inblock = self.inBlock1(inp_all)
            else:
                input_clear = F.interpolate(
                    clear_features, (hi, wi), mode="bilinear"
                ).to(input.device)
                input_pred = F.interpolate(
                    input_pre, (hi, wi), mode="bilinear"
                ).to(input.device)
                inp_all = torch.cat((input_clear, input_pred), 1)
                first_scale_inblock = self.inBlock2(inp_all)

            first_scale_encoder_first = self.encoder_first(first_scale_inblock)

            first_scale_encoder_second = self.encoder_second(
                first_scale_encoder_first
            )
            first_scale_decoder_second = self.decoder_second(
                first_scale_encoder_second
            )

            first_scale_decoder_first = self.decoder_first(
                first_scale_decoder_second + first_scale_encoder_first
            )
            input_pre = self.outBlock(
                first_scale_decoder_first + first_scale_inblock
            )
            out = self.outBlock2(input_pre)
            output.append(out)

        return output
