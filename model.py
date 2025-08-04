import torch.nn as nn
from scipy.optimize import minimize_scalar
from numpy.linalg import cholesky
import numpy as np

class HEstimatorCNN(nn.Module):
    def __init__(self,seq_len = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=15,padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(32,64,kernel_size=15,padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self,x):
        return self.net(x.unsqueeze(1))
    
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
            nn.Linear(256,1)
        )

    def forward(self,x):
        return self.net(x)
    
def construct_cov(num_steps, H, dt):
    cov = np.zeros((num_steps, num_steps))
    # now assume observations at t = dt, 2dt, …, num_steps·dt
    for i in range(num_steps):
        for j in range(num_steps):
            s = (i + 1) * dt
            t = (j + 1) * dt
            cov[i, j] = 0.5 * (t**(2 * H) + s**(2 * H) - abs(t - s)**(2 * H))
    return cov


def neg_log_likelihood(num_steps: int, Hurst_exp: float, X,time_interval):
    R = construct_cov(num_steps,Hurst_exp,time_interval)
    L = np.linalg.cholesky(R)
    log_det_R = 2*np.sum(np.log(np.diag(L)))
    quad = X.T @ np.linalg.solve(R,X)
    l = .5*num_steps*np.log(quad)+.5*log_det_R
    return l 



def estimate_H_mle_from_fBM(fBM,dt):
    n = int(len(fBM))
    result = minimize_scalar(
            fun = lambda H: neg_log_likelihood(n,H,fBM,dt),
            method = "bounded",
            bounds = (1e-5,1-1e-5),
            options={
                "xatol":1e-6,
                "maxiter":100
                }

            )
    H_hat = result.x
    R = construct_cov(n,H_hat,dt)
    v_hat = np.sqrt((fBM.T @ np.linalg.solve(R,fBM))/n)
    return H_hat,v_hat