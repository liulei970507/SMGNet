import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from typing import Optional, Any, Tuple
import numpy as np

class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)

class weight_prediction(nn.Module):
    def __init__(self, dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gmp = nn.AdaptiveMaxPool2d(1)
        self.weight_pred = nn.Sequential(GRL(),
                                         nn.Linear(dim*4,dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim,dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim,1))
        # nn.Sequential(nn.Conv2d(dim*2,dim//32,1,bias=True),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(dim//32,1,1,bias=True))
                                        #  nn.Sigmoid())
        
        

    def forward(self, x_v, x_i):
        B, N, C = x_v.shape

        z_v = x_v[:,:-self.patch_size**2,:]
        x_v = x_v[:,-self.patch_size**2:,:]
        z_i = x_i[:,:-self.patch_size**2,:]
        x_i = x_i[:,-self.patch_size**2:,:]

        z_v = token2feature(z_v)
        x_v = token2feature(x_v)
        z_i = token2feature(z_i)
        x_i = token2feature(x_i)

        # z = torch.cat([self.gap(z_v), self.gap(z_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
        # x = torch.cat([self.gap(x_v), self.gap(x_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
        # z = self.weight_pred(z)
        # x = self.weight_pred(x)
        x_v = torch.cat([self.gap(z_v), self.gap(x_v)],dim=1).reshape(B, -1)
        x_i = torch.cat([self.gap(z_i), self.gap(x_i)],dim=1).reshape(B, -1)
        x = torch.cat([x_v, x_i], dim=1)
        x = self.weight_pred(x)
        return x.reshape(B)

class weight_prediction_mid(nn.Module):
    def __init__(self, dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gmp = nn.AdaptiveMaxPool2d(1)
        self.weight_pred = nn.Sequential(nn.Linear(dim*2,dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim,dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(dim,1))
        # nn.Sequential(nn.Conv2d(dim*2,dim//32,1,bias=True),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(dim//32,1,1,bias=True))
                                        #  nn.Sigmoid())
        
        

    def forward(self, x_v, x_i):
        B, N, C = x_v.shape
        # x_v = x_v[:, -self.patch_size**2:, :].transpose(-1,-2).reshape(B,-1,self.patch_size,self.patch_size).contiguous()
        # x_i = x_i[:, -self.patch_size**2:, :].transpose(-1,-2).reshape(B,-1,self.patch_size,self.patch_size).contiguous()
        z_v = x_v[:,:-self.patch_size**2,:]
        x_v = x_v[:,-self.patch_size**2:,:]
        z_i = x_i[:,:-self.patch_size**2,:]
        x_i = x_i[:,-self.patch_size**2:,:]

        z_v = token2feature(z_v)
        x_v = token2feature(x_v)
        z_i = token2feature(z_i)
        x_i = token2feature(x_i)

        z = torch.cat([self.gap(z_v), self.gap(z_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
        x = torch.cat([self.gap(x_v), self.gap(x_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
        z = self.weight_pred(z)
        x = self.weight_pred(x)

        return torch.stack([z.reshape(B),x.reshape(B)],dim=1)

class weight_prediction_tiny(nn.Module):
    def __init__(self, dim=768, patch_size=16,N=320):
        super().__init__()
        self.patch_size = patch_size
        # 256(16*16)x768xb 64(8*8)x768xb
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gmp = nn.AdaptiveMaxPool2d(1)
        # self.weight_pred = nn.Sequential(nn.Linear(dim*2,dim//2),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(dim//2,dim//2),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(dim//2,1))
        # nn.Sequential(nn.Conv2d(dim*2,dim//32,1,bias=True),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(dim//32,1,1,bias=True))
                                        #  nn.Sigmoid())
        # self.weight_pred = nn.Sequential(nn.Linear(N*2,N*4),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(N*4,1))

        self.fc1 = nn.Linear(N*2,N)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(N,N)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(N,1)
        # self.drop3 = nn.Dropout(0.1)
        

    def forward(self, x_v, x_i):
        B,N,C=x_v.shape
        x = torch.cat([x_v,x_i],dim=1)
        x = torch.mean(x,dim=2)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        # x = self.drop3(x)
        return x.reshape(B)
    # def forward(self, x_v, x_i):
    #     B, N, C = x_v.shape
    #     # x_v = x_v[:, -self.patch_size**2:, :].transpose(-1,-2).reshape(B,-1,self.patch_size,self.patch_size).contiguous()
    #     # x_i = x_i[:, -self.patch_size**2:, :].transpose(-1,-2).reshape(B,-1,self.patch_size,self.patch_size).contiguous()
    #     # z_v = x_v[:,:-self.patch_size**2,:]
    #     # x_v = x_v[:,-self.patch_size**2:,:]
    #     # z_i = x_i[:,:-self.patch_size**2,:]
    #     # x_i = x_i[:,-self.patch_size**2:,:]

    #     # z_v = token2feature(z_v)
    #     # x_v = token2feature(x_v)
    #     # z_i = token2feature(z_i)
    #     # x_i = token2feature(x_i)

    #     # z = torch.cat([self.gap(z_v), self.gap(z_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
    #     # x = torch.cat([self.gap(x_v), self.gap(x_i)],dim=1).permute(0,2,3,1).reshape(B, -1)
    #     # z = self.weight_pred(z)
    #     # x = self.weight_pred(x)
    #     x = torch.cat([x_v,x_i],dim=1)
    #     x = torch.mean(x,dim=2)
    #     return torch.stack([z.reshape(B),x.reshape(B)],dim=1)
    
def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

class weight_prediction_sigmoid(nn.Module):
    def __init__(self, dim=768, patch_size=16,N=320):
        super().__init__()
        self.patch_size = patch_size
        
        self.fc1 = nn.Linear(N*2,N)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(N,N)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(N,1)
        # self.drop3 = nn.Dropout(0.1)
        

    def forward(self, x_v, x_i):
        B,N,C=x_v.shape
        x = torch.cat([x_v,x_i],dim=1)
        x = torch.mean(x,dim=2)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        # x = self.drop3(x)
        return x.reshape(B)
    

class MS_Sigmoid(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x_search):
        # x_templeate = self.gap(x_templeate).view(x_templeate.size(0), -1)
        x = self.gap(x_search).view(x_search.size(0), -1)
        # x = torch.cat([x_templeate,x_search],dim=1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # x = F.sigmoid(x)
        return x
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x_search):
        # x_templeate = self.gap(x_templeate).view(x_templeate.size(0), -1)
        x = self.gap(x_search).view(x_search.size(0), -1)
        # x = torch.cat([x_templeate,x_search],dim=1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = F.sigmoid(x)
        return x