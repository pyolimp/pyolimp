from . import (
    PatchEmbed,
    BasicLayer,
    PatchMerging,
    Upsample_promotion,
    Upsample,
    Upsample_layer,
    resi_connection_layer,
)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from ..download_path import download_path, PyOlimpHF


class Generator_transformer_pathch4_6_3_48_3(nn.Module):
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
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[6, 6, 6],
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

        transform_val_list1 = [
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        ]
        transform_val_list2 = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.trans_compose1 = transforms.Compose(transform_val_list1)
        self.trans_compose2 = transforms.Compose(transform_val_list2)
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
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim=embed_dim * 2,
                output_dim=embed_dim,
                norm_layer=norm_layer,
            ),
            Upsample_layer(
                input_resolution=(
                    patches_resolution[0] * 2,
                    patches_resolution[1] * 2,
                ),
                dim=embed_dim,
                output_dim=embed_dim // 2,
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
            nn.Conv2d(embed_dim // 2, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
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

    def forward_features(self, x):
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
                x1 = torch.cat(
                    (
                        x1,
                        self.resi_connection[1 - i](
                            self.downsample_result[1 - i]
                        ),
                    ),
                    -1,
                )
            i = i + 1

        x = x1
        x = self.final_upsample(x)
        x = x.permute(0, 2, 1)  # B C ,H*W
        x = x.view([-1, 48, 256, 256])
        x = self.final(x)
        return x

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.trans_compose2(tensor)

    def postprocess(self, tensors: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        return (self.trans_compose1(tensors[0]),)

    def arguments(self, *args, **kwargs):
        return {}

    @classmethod
    def from_path(
        cls,
        path: PyOlimpHF = "hf://CVD/cvd_swin.pth",
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

    def forward(self, x):
        x = self.forward_features(x)
        return (x,)

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

    swd_swin = Generator_transformer_pathch4_6_3_48_3()
    output = swd_swin(test_t)

    plt.imshow(output.detach().cpu().numpy().transpose([0, 2, 3, 1])[0])
    plt.savefig("fig3.png")
    plt.show()
