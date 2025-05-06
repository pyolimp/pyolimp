from . import (
    PatchEmbed,
    BasicLayer,
    PatchMerging,
    Upsample_promotion,
    Upsample,
    resi_connection_layer,
)
from math import sqrt
from torch import Tensor, tensor
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from ..download_path import download_path, PyOlimpHF
from olimp.evaluation.cs.linrgb import linRGB
from olimp.evaluation.cs.srgb import sRGB
from olimp.evaluation.cs.lms import LMS


class CVDSwin4Channels(nn.Module):
    r"""Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=4,
        num_classes=1000,
        embed_dim=96,
        depths=[8, 4, 4],
        num_heads=[8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.simple_conv unused, but needed for old weights (else error)
        self.simple_conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            if i_layer == 0:
                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample_promotion,
                    use_checkpoint=use_checkpoint,
                )
            else:
                layer = BasicLayer(
                    dim=int(
                        embed_dim * 2 ** (self.num_layers - 1 - i_layer) * 2
                    ),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[self.num_layers - i_layer - 1],
                    num_heads=num_heads[self.num_layers - i_layer - 1],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=Upsample,
                    use_checkpoint=use_checkpoint,
                )
            self.uplayers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.resi_connection = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = resi_connection_layer(
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                dim=int(embed_dim * 2**i_layer),
                output_dim=int(embed_dim * 2**i_layer),
            )
            self.resi_connection.append(layer)

        self.flinal_layer = nn.Sequential(
            # resi_connection_layer(embed_dim*)
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                embed_dim * 2,
                embed_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                norm_layer=norm_layer,
            ),
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=3, padding=1, stride=1),
            nn.Softplus(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            x = layer(x)

            self.downsample_result.append(x)
        i = 0
        x1 = x

        for uplayer in self.uplayers:
            x1 = uplayer(x1)
            if i < 3:
                # x1 = torch.cat((x1, self.resi_connection[1-i](self.downsample_result[1-i])), -1)
                x1 = torch.cat((x1, self.downsample_result[1 - i]), -1)
            i = i + 1

        x = x1

        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        return x

    def forward(self, inputs) -> Tensor:
        from olimp.processing import quantile_clip

        x, hue_angle_deg = inputs

        x1 = self._add_cvd_dim(x, hue_angle_deg)  # 4 channels
        x1 = self.forward_features(x1)  # output: 1 channel
        x1 = quantile_clip(x * x1 + 1e-5, quantile=0.70)

        return (x1,)

    @classmethod
    def from_path(
        cls,
        path: PyOlimpHF = "hf://CVD/cvd_swin_4channels.pth",
    ):
        path = download_path(path)
        state_dict = torch.load(
            path,
            map_location=torch.get_default_device(),
            weights_only=True,
        )
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model = cls()
        model.load_state_dict(new_state_dict)
        return model

    @staticmethod
    def _srgb2linRGB(srgb: Tensor) -> Tensor:
        output = []
        for i in srgb:
            output.append(linRGB().from_sRGB(i))
        return torch.stack(output, dim=0)

    @staticmethod
    def _linRGB2srgb(linRGB: Tensor) -> Tensor:
        output = []
        for i in linRGB:
            output.append(sRGB().from_linRGB(i))
        return torch.stack(output, dim=0)

    @staticmethod
    def _add_cvd_dim(linRGB_batch: Tensor, hue_angle_deg: Tensor):
        # broadcasted matrix [3x3]
        sqrt3 = sqrt(3.0)
        lms_vec1 = tensor(
            (
                (2, 0, 1),
                (-1, sqrt3, 1),
                (-1, -sqrt3, 1),
            )
        )

        # hue vector [B, 3, 1]
        lms_vec2 = torch.ones((len(hue_angle_deg), 3, 1))
        lms_vec2[:, 0, 0] = torch.cos(torch.deg2rad(hue_angle_deg))
        lms_vec2[:, 1, 0] = torch.sin(torch.deg2rad(hue_angle_deg))

        # dot product [B, 3, 1]
        lms_vec = torch.matmul(lms_vec1, lms_vec2) / 3

        def _lms2linRGB(lms: Tensor) -> Tensor:
            output = []
            for i in lms:
                output.append(linRGB().from_XYZ(LMS().to_XYZ(i)))
            return torch.stack(output, dim=0)

        linRGB_vec = _lms2linRGB(lms_vec)

        # make fourth dim (dot product of img and linRGB_vec); maybe better way?
        output = []
        for i in range(linRGB_batch.shape[0]):
            output.append(
                torch.tensordot(
                    linRGB_batch[i].permute(1, 2, 0), linRGB_vec[i], dims=1
                ).permute(2, 0, 1)
            )  # rgb * cvd vector dot
        dim4 = torch.stack(output)
        return torch.hstack((linRGB_batch, dim4))

    def preprocess(self, tensor: Tensor, hue_angle_deg: Tensor) -> Tensor:
        return self._srgb2linRGB(tensor), hue_angle_deg

    def postprocess(self, tensor: Tensor) -> Tensor:
        return self._linRGB2srgb(tensor)

    def arguments(self, *args, **kwargs):
        return {}


def _demo():
    from ...._demo_cvd import demo
    from typing import Callable
    from olimp.simulate.color_blindness_distortion import (
        ColorBlindnessDistortion,
    )

    def demo_cvd_swin(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        svd_swin = CVDSwin4Channels.from_path()
        image = svd_swin.preprocess(image)
        progress(0.1)
        precompensation = svd_swin(image)
        progress(1.0)
        return (svd_swin.postprocess(precompensation[0]),)

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo(
        "CVD-SWIN",
        demo_cvd_swin,
        distortion=distortion,
    )


if __name__ == "__main__":
    _demo()
