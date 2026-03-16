from functools import partial
from turtle import forward
from timm.models.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock

class QATCLSWGTELayer(nn.Module):
    def __init__(self, MLP, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.mlp = MLP
        self.patch_size = 16
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rgb_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.tir_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.rgb2tir = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.tir2rgb = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        
    def forward(self, x_v, x_i):
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        
        x = torch.cat([x_v, x_i], dim=1)
        enc_opt1 = x[:, lens_z:lens_z + lens_x, :]
        enc_opt2 = x[:, -lens_x:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, 16, 16)
        list_weight = [None]*len(self.mlp)
        for idx in range(len(self.mlp)):
            list_weight[idx] = self.mlp[idx](opt_feat)
        tensor_weight = torch.stack(list_weight,1)
        weight = tensor_weight.sigmoid().mean(dim=1)

        x_v_clone = x_v[:, lens_z:, :].clone()
        x_i_clone = x_i[:, lens_z:, :].clone()
        
        x_v[:, lens_z:, :] = self.tir2rgb(torch.cat([x_v_clone, (1-weight.unsqueeze(-1))*self.tir_encoder(x_i_clone)], dim=1))[:, :lens_x, :]
        x_i[:, lens_z:, :] = self.rgb2tir(torch.cat([x_i_clone, weight.unsqueeze(-1)*self.rgb_encoder(x_v_clone)], dim=1))[:, :lens_x, :]
        # woweight
        # print('woweight')
        # x_v[:, lens_z:, :] = self.tir2rgb(torch.cat([x_v_clone, self.tir_encoder(x_i_clone)], dim=1))[:, :lens_x, :]
        # x_i[:, lens_z:, :] = self.rgb2tir(torch.cat([x_i_clone, self.rgb_encoder(x_v_clone)], dim=1))[:, :lens_x, :]
        
        return x_v, x_i, tensor_weight
