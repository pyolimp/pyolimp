from . import (
    PatchEmbed,
    BasicLayer,
    PatchMerging,
    Upsample_promotion,
    Upsample,
    resi_connection_layer,
)

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Generator_transformer_pathch2_no_Unt(nn.Module):
    r"""patch size is 2."""

    r""" Swin Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 2
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
        patch_size=2,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[4, 6],
        num_heads=[3, 6],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes
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
            if i_layer != -1:

                layer = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
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
                    downsample=Upsample_promotion,
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

        self.apply(self._init_weights)

        self.final_upsample = nn.Sequential(
            Upsample_promotion(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim,
                norm_layer=norm_layer,
            ),
            # Upsample_promotion(input_resolution=(patches_resolution[0] * 2, patches_resolution[1] * 2),
            #                    dim=int(embed_dim // 2), norm_layer=norm_layer),
        )

        self.final = nn.Sequential(
            # Upsample(x
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(embed_dim, 3, 4, padding=1),
            # nn.Linear(int(embed_dim //2), 3, bias=False),
            # nn.Conv2d(int(embed_dim/2), 3, 3, padding=1),
            # nn.Conv2d()
            # nn.ConvTranspose2d(embed_dim/4, 3, 4, 2, 1, bias=False),
            nn.Conv2d(int(embed_dim // 2), 3, 1, padding=0),
            nn.Tanh(),
        )

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

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        self.downsample_result = [x]
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
            self.downsample_result.append(x)
        i = 0

        for xx in self.downsample_result:
            i = i + 1
        x1 = x

        i = 0
        for uplayer in self.uplayers:

            x1 = uplayer(x1)
            # print(x1.size())
            # if i < 1:
            #     x1 = torch.cat((x1, self.downsample_result[0-i]), -1)
            i = i + 1

        x = x1

        # x = self.norm(x)  # B L C
        # print(x.size())
        x = self.final_upsample(x)

        # print(x.size())
        # print(x1123)

        # x = x.view(-1, C, H * W)
        x = x.permute(0, 2, 1)  # B C ,H*W
        # print(x.size())
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)

        # x = self.final(x)

        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print('forward',x.size())
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    test_data = np.load(
        "/home/devel/olimp/pyolimp/tests/test_data/test.npy", allow_pickle=True
    )

    test = test_data[3]
    test = test.clip(0, 1)
    test_t = torch.tensor(test).unsqueeze(0)

    swd_swin = Generator_transformer_pathch2_no_Unt()
    output = swd_swin(test_t)

    plt.imshow(output.detach().cpu().numpy().transpose([0, 2, 3, 1])[0])
    plt.savefig("fig2.png")
    plt.show()
