import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from numpy.linalg import cholesky
import numpy as np
from data_simulator import construct_cov, simulate_S
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


class HEstimatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(1, affine=False)
        self.net = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=7,padding=3,dilation=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4,stride=4),  # downsample for speed (set to 1 if you want full res)

    # Dilated stack to see long horizons without blowing up compute
            nn.Conv1d(32,32,kernel_size=7,padding=3,dilation=1),
            nn.GroupNorm(1,32),
            nn.ReLU(),

            nn.Conv1d(32,32,kernel_size=7,padding=6,dilation=2),
            nn.GroupNorm(1,32),
            nn.ReLU(),

            nn.Conv1d(32,32,kernel_size=7,padding=12,dilation=4),
            nn.GroupNorm(1,32),
            nn.ReLU(),

            nn.Conv1d(32,32,kernel_size=7,padding=24,dilation=8),
            nn.GroupNorm(1,32),
            nn.ReLU(),

            nn.Conv1d(32,32,kernel_size=7,padding=48,dilation=16),
            nn.GroupNorm(1,32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
            )
        # self.price_to_vol = nn.Sequential(
        #     nn.Conv1d(1,32,kernel_size = 7,padding=3,dilation=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32,64,kernel_size = 7,padding=3,dilation=1),
        #     nn.ReLU(),
        #     nn.Conv1d(64,64,kernel_size = 3,padding=1,stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(64,64,kernel_size = 3,padding=1,stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(64,16,kernel_size=1)
        # )



        # self.vol_to_H = nn.Sequential(
        #     nn.Conv1d(16,64,kernel_size=15,padding=7,dilation=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64,64,kernel_size=15,padding=14,dilation=2),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64,64,kernel_size = 15,padding=28,dilation=4),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=15, padding=56, dilation=8),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(64,128),
        #     nn.ReLU(),
        #     nn.Linear(128,1),
        #     nn.Sigmoid()
        # )

    
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.inorm(x)
        x = self.net(x)
        #x = self.price_to_vol(x)
        #return self.vol_to_H(x)
        return x
    
class HEstimatorFNN(nn.Module):
    def __init__(self,input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)
    

class FixedDiff(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor([[[1.,-1.]]])
        self.conv = nn.Conv1d(1,1,kernel_size = 2,padding=1,bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(w)
        for p in self.conv.parameters():
            p.requires_grad=False
    def forward(self,x):
        y = self.conv(x)
        return y[...,:x.size(-1)]
    
class ResBlock(nn.Module):
    def __init__(self,c,dilation):
        super().__init__()
        k=7
        pad = (k-1)//2*dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(c,c,k,padding=pad,dilation=dilation))
        self.gn1 = nn.GroupNorm(8,c)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(c,c,k,padding=pad,dilation=dilation))
        self.gn2 = nn.GroupNorm(8,c)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c,c//8,1),
            nn.GELU(),
            nn.Conv1d(c//8,c,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = h*self.se(h)
        return F.gelu(x+h)
        
class HurstNet(nn.Module):
    def __init__(self,base=32,blocks = (1,2,4,8)):
        super().__init__()
        self.diff = FixedDiff()
        self.stem = nn.Conv1d(1,base,15,padding=7)
        self.gn0 = nn.GroupNorm(8,base)
        self.tcn = nn.Sequential(
            *[ResBlock(base,d) for d in blocks]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base,128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):

        if x.dim() == 2:  
            x = x.unsqueeze(1)           # (B, L) -> (B, 1, L)
        elif x.size(1) != 1:              # (B, C, L) but C != 1
            x = x.mean(dim=1, keepdim=True) 


        x = self.diff(x)
        x = F.gelu(self.gn0(self.stem(x)))
        x = self.tcn(x)
        h = self.head(x)
        
        return h




def neg_log_likelihood(num_steps: int, Hurst_exp: float, X,time_interval):
    R = construct_cov(num_steps,Hurst_exp,time_interval)
    L = np.linalg.cholesky(R)
    log_det_R = 2*np.sum(np.log(np.diag(L)))
    quad = X[1:].T @ np.linalg.solve(R,X[1:])
    l = .5*num_steps*np.log(quad)+.5*log_det_R
    return l 

def H_mle_fbm(fbm,T):
    n = len(fbm)
    dt = T/n
    result = minimize_scalar(
        fun=lambda H: neg_log_likelihood(n,H,fbm,dt),
        method = "bounded",
        bounds = (1e-4,1-1e-4),
        options = {
            "xatol":1e-6,
            "maxiter":100
        }
    )
    H_hat = result.x
    return H_hat

def calc_spot_var(S,m):
    log_returns = np.log(S[1:]/S[:-1])
    num_obs = len(log_returns)//m
    spot_vars = np.zeros(num_obs)
    for i in range(0,num_obs):
        spot_vars[i] = np.mean(log_returns[i*m:i*m+m]**2)
    return spot_vars

def mle_from_stock(S,T,grad):
    spot_var = calc_spot_var(S,grad)
    X = np.log(spot_var)
    fbm = X-X.mean()
    H_hat = H_mle_fbm(fbm,T)
    return H_hat
