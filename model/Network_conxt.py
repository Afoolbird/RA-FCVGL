# # -*- coding: utf-8 -*-
# # Citation:
# # @article{qu2021transmef,
# #   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
# #   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
# #   journal={arXiv preprint arXiv:2112.01030},
# #   year={2021}
# # }
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum 
# from einops import rearrange
# from einops.layers.torch import Rearrange
# from .VGG16 import VGG16



# def save_grad(grads, name):
#     def hook(grad):
#         grads[name] = grad

#     return hook


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class OutConv(nn.Module):
#     """1*1 conv before the output"""

#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# # class Encoder(nn.Module):
# #     """features extraction"""

# #     def __init__(self):
# #         super(Encoder, self).__init__()
# #         self.vgg = DoubleConv(1, 16)
# #         self.layer1 = DoubleConv(16, 32) 
# #         self.layer2 = DoubleConv(32, 48)

# #     def forward(self, x, grads=None, name=None):
# #         x = self.vgg(x)
# #         x = self.layer1(x)
# #         x = self.layer2(x)

# #         if grads is not None:
# #             x.register_hook(save_grad(grads, name + "_x"))
# #         return x


# # class Encoder_Trans(nn.Module):
# #     """features extraction"""

# #     def __init__(self):
# #         super(Encoder_Trans, self).__init__()
# #         self.vgg = DoubleConv(1, 16)
# #         self.layer1 = DoubleConv(17, 32)
# #         self.layer2 = DoubleConv(32, 48)
# #         self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
# #                                emb_dropout=0.1)

# #     def forward(self, x, grads=None, name=None):
# #         x_e = self.vgg(x)
# #         x_t = self.transformer(x)
# #         x = torch.cat((x_e, x_t), dim=1)
# #         x = self.layer1(x)
# #         x = self.layer2(x)

# #         if grads is not None:
# #             x.register_hook(save_grad(grads, name + "_x"))
# #         return x

# # class Encoder_Trans(nn.Module):
# #     """features extraction"""

# #     def __init__(self,image_size):
# #         super(Encoder_Trans, self).__init__()
# #         self.image_size = image_size
# #         self.vgg = VGG16()                                       #DoubleConv(1, 16)
# #         self.layer1 = DoubleConv(513, 1024)
# #         self.layer2 = DoubleConv(1024, 512)
# #         self.transformer = ViT(image_size=self.image_size, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
# #                                emb_dropout=0.1)

# #     def forward(self, x, grads=None, name=None):
# #         x_e, sat_320, sat_160, sat_80, sat_40, sat_20 = self.vgg(x)
# #         x_t = self.transformer(x)
# #         x = torch.cat((x_e, x_t), dim=1)
# #         x = self.layer1(x)
# #         x = self.layer2(x)

# #         if grads is not None:
# #             x.register_hook(save_grad(grads, name + "_x"))
# #         return x, sat_320, sat_160, sat_80, sat_40, sat_20
    
# # class Decoder(nn.Module):
# #     """reconstruction"""

# #     def __init__(self):
# #         super(Decoder, self).__init__()
# #         self.layer1 = DoubleConv(48, 32)
# #         self.layer2 = DoubleConv(32, 16)
# #         self.outc = OutConv(16, 1)

# #     def forward(self, x):
# #         x = self.layer1(x)
# #         x = self.layer2(x)
# #         output = self.outc(x)
# #         return output


# # class Decoder_Trans(nn.Module):
# #     """reconstruction"""

# #     def __init__(self):
# #         super(Decoder_Trans, self).__init__()
# #         self.layer3 = DoubleConv(49, 48)
# #         self.layer4 = DoubleConv(48, 48)
# #         self.layer1 = DoubleConv(48, 32)
# #         self.layer2 = DoubleConv(32, 16)
# #         self.outc = OutConv(16, 1)

# #     def forward(self, x):
# #         x = self.layer4(self.layer3(x))
# #         x = self.layer1(x)
# #         x = self.layer2(x)
# #         output = self.outc(x)
# #         return output


# # class SimNet(nn.Module):
# #     """easy network for self-reconstruction task"""

# #     def __init__(self):
# #         super(SimNet, self).__init__()
# #         self.encoder = Encoder()
# #         self.decoder = Decoder()

# #     def forward(self, x):
# #         x = self.encoder(x)
# #         x = self.decoder(x)
# #         return x


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)




# class Transformer_sat(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))

#     def forward(self, x):
#         x_trans_features=[]
#         x_trans_features_=[]
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#             x_trans_features.append(x)
#             # print('Tramsformer local features shape:',x.shape)
#         x_trans_features_.append(x_trans_features[1])
#         x_trans_features_.append(x_trans_features[3])
#         x_trans_features_.append(x_trans_features[5])
#         x_trans_features_.append(x_trans_features[7])
#         x_trans_features_.append(x_trans_features[9])
#         x_trans_features_.append(x_trans_features[11])
#         return x_trans_features_
    
# class ViT_sat(nn.Module):
#     def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64,
#                  dropout=0., emb_dropout=0.):
#         super().__init__()
#         assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

#         patch_dim = channels * patch_size ** 2

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
#             nn.Linear(patch_dim, dim)
#         )
#         self.image_size = image_size
#         self.dim = dim
#         self.patch_size = patch_size
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer_sat(dim, depth, heads, dim_head, mlp_dim, dropout)

#         # self.convd1 = nn.Sequential(
#         #     nn.Conv2d(3, 3, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.BatchNorm2d(3))
#         self.convd_11 = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(512)
#             )
#         self.convd_9 = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(256)
#             )
#         self.convd_7 = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(128)
#             )
#         self.convd_5 = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(64)
#             )
#         self.convd_3 = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(32)
#             )
#         self.convd_1 = nn.Sequential(
#         nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(16)
#             ) 
#     def forward(self, img):
#         x = self.to_patch_embedding(img)  # [B,256,256]
#         b, n, _ = x.shape

#         # cls_tokens = self.cls_token.expand(b, -1, -1)
#         # x = torch.cat((cls_tokens, x), dim=1)
#         # x = self.dropout(x)


#         x_fwatures = self.transformer(x)
#         # print('x.shape:',x.shape)
#         h = self.image_size[0] // self.patch_size
#         w = self.image_size[1] // self.patch_size
#         # x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=h, w=w, c=1)(x)  # [B,1,320,320]
#         # print('x.shape:',x.shape)
#         for i in range(len(x_fwatures)):
#             x_fwatures[i]=Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=h, w=w, c=1)(x_fwatures[i])
#         x_fwatures[0]=self.convd_1(x_fwatures[0])
#         x_fwatures[1]=self.convd_3(x_fwatures[1])
#         x_fwatures[2]=self.convd_5(x_fwatures[2])
#         x_fwatures[3]=self.convd_7(x_fwatures[3])
#         x_fwatures[4]=self.convd_9(x_fwatures[4])
#         x_fwatures[5]=self.convd_11(x_fwatures[5])
#         # print('x.shape:',x.shape)

#         return x_fwatures


# class Transformer_grd(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#             # print('Tramsformer local features shape:',x.shape)
#         return x
    
# class ViT_grd(nn.Module):
#     def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64,
#                  dropout=0., emb_dropout=0.):
#         super().__init__()
#         assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

#         patch_dim = channels * patch_size ** 2

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
#             nn.Linear(patch_dim, dim)
#         )
#         self.image_size = image_size
#         self.dim = dim
#         self.patch_size = patch_size
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer_grd(dim, depth, heads, dim_head, mlp_dim, dropout)

#         # self.convd1 = nn.Sequential(
#         #     nn.Conv2d(3, 3, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.BatchNorm2d(3))
#         self.convd = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(512)
#             )
#     def forward(self, img):
#         x = self.to_patch_embedding(img)  # [B,256,256]
#         b, n, _ = x.shape

#         # cls_tokens = self.cls_token.expand(b, -1, -1)
#         # x = torch.cat((cls_tokens, x), dim=1)
#         # x = self.dropout(x)


#         x = self.transformer(x)
#         # print('x.shape:',x.shape)
#         h = self.image_size[0] // self.patch_size
#         w = self.image_size[1] // self.patch_size
#         x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=h, w=w, c=1)(x)  # [B,3,320,320]
#         # print('x.shape:',x.shape)
#         x=self.convd(x)
#         # print('x.shape:',x.shape)

#         return x

# # class TransNet(nn.Module):
# #     """U-based network for self-reconstruction task"""

# #     def __init__(self, image_size=None):
# #         super(TransNet, self).__init__()

# #         self.encoder = Encoder_Trans(image_size)
# #         # self.decoder = Decoder()

# #     def forward(self, x):
# #         x, x_5, x_4, x_3, x_2, x_1 = self.encoder(x)
# #         # x = self.decoder(x)
# #         return x, x_5, x_4, x_3, x_2, x_1

#######################################################################################################################################



# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

        self.convd_320 = nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(24, 3, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        LayerNorm(3, eps=1e-6, data_format="channels_first"),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.convd_320(x)
        x_features=[]
        x_features.append(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            # print(f'Input features shape of downsample_layer{i}:',x.shape)
            x = self.stages[i](x)
            # print(f'Input features shape of stage{i}:',x.shape)
            x_features.append(x)
        
        # Implementing Feature Recombination Module
        # It's so simple and requires only a few lines of code!
        if x.shape[-1] == x.shape[-2]:
            x = torch.cat((self.norm(x[:, :, x.shape[-2] // 2:, :x.shape[-1] // 2].mean([-2, -1])), \
                           self.norm(x[:, :, :x.shape[-2] // 2, :x.shape[-1] // 2].mean([-2, -1])), \
                           self.norm(x[:, :, :x.shape[-2] // 2:, x.shape[-1] // 2:].mean([-2, -1])), \
                           self.norm(x[:, :, x.shape[-2] // 2:, x.shape[-1] // 2:].mean([-2, -1]))), dim=1)
            # print('x_sat features shape:',x.shape)
        else:
            x = torch.cat((self.norm(x[:, :, :, :x.shape[-1] // 4].mean([-2, -1])), \
                           self.norm(x[:, :, :, x.shape[-1] // 4: x.shape[-1] // 4 * 2].mean([-2, -1])), \
                           self.norm(x[:, :, :, x.shape[-1] // 4 * 2: x.shape[-1] // 4 * 3].mean([-2, -1])), \
                           self.norm(x[:, :, :, x.shape[-1] // 4 * 3:].mean([-2, -1]))), dim=1)
            # print('x_grd features shape:',x.shape)
        return x_features, x

    def forward(self, x):
        x_features, x = self.forward_features(x)
        # x = self.head(x)
        return x_features, x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        print('Model Convnext_tiny is loaded!')
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
