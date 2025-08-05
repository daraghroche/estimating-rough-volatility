import numpy as np
from scipy.fftpack import fft, ifft
import torch
from torch.utils.data import Dataset
from numpy.linalg import cholesky

def simulate_fbm_increments(H: float, T: float, n: int) -> np.ndarray:

    # Build circulant covariance matrix
    dt = T/n
    k = np.arange(0,n+1)
    r = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + np.abs(k-1)**(2*H))
    r_aug = np.concatenate([r,r[-2:0:-1]])
    # FFT based embedding
    lam = np.real(fft(r_aug))
    if np.any(lam<0):
        raise RuntimeError("Negative eigenvalue in circulant embedding")
    W = np.sqrt(lam/(2*n)) * (np.random.randn(len(r_aug)) + 1j*np.random.randn(len(r_aug)))
    Z = fft(W)
    fgn = np.real(Z[:n+1])
    fbm = np.cumsum(fgn)
    increments = np.diff(fbm)
    increments *=dt**H
    return increments

# def construct_cov(num_steps, H, T):
#     dt = T/(num_steps-1)
#     cov = np.zeros((num_steps-1, num_steps-1))
#     for i in range(num_steps-1):
#         for j in range(num_steps-1):
#             s = (i + 1) * dt
#             t = (j + 1) * dt
#             cov[i, j] = 0.5 * (t**(2 * H) + s**(2 * H) - abs(t - s)**(2 * H))
#     return cov

def construct_cov(num_steps,H,T):
    steps = np.arange(1,num_steps,1)
    dt = T/(num_steps)
    t = steps*dt
    t_i = t[:,None]
    t_j = t[None,:]
    cov = 0.5 * (t_i**(2*H) + t_j**(2*H)-np.abs(t_i-t_j)**(2*H))
    return cov


def simulate_S(num_steps,T,L):
    dt = T/num_steps

    Z = np.random.randn(num_steps-1)
    BH = L @ Z
    BH = np.concatenate([[0],BH])
    dB = Z*np.sqrt(dt)
    B = np.concatenate([[0],np.cumsum(dB)])
    dW = np.random.randn(num_steps-1)*np.sqrt(dt)
    W = np.concatenate([[0],np.cumsum(dW)])
    
    # Fixed params
    nu = 1
    S0 = 1
    p = -.65
    pbar = np.sqrt(1-p**2)
    V0 = 0.1

    V = V0*np.exp(nu*BH)
    dZ = p *dB + pbar*dW
    log_returns = -.5*V[:-1] *dt+np.sqrt(V[:-1])*dZ
    log_S = np.cumsum(log_returns)
    S = S0 *np.exp(np.insert(log_S,0,0))
    return S


def add_noise_and_jumps(
        increments: np.ndarray,
        sigma_micro: float = 0.01,
        jump_rate: float = 0.0,
        jump_size: float = 0.1
) -> np.ndarray:
    
    n = len(increments)
    noise = sigma_micro * np.random.randn(n)
    jumps = np.zeros(n)
    num_jumps = np.random.poisson(jump_rate)
    if num_jumps>0:
        jump_times = np.random.choice(n,size = num_jumps,replace = False)
        jumps[jump_times] = jump_size * np.random.choice([-1,1],size=num_jumps)
    return increments + noise + jumps



# if __name__ == "__main__":
#     data = []
#     for _ in range(100):
#         inc = simulate_fbm(H=0.5,T=1.0,n=1024)
#         inc_noisy = add_noise_and_jumps(inc,sigma_micro=0.02,jump_rate=0.1,jump_size=0.1)
#         data.append((inc_noisy,0.05))
#     dataset = FBMDataset(data)
#     print("Dataset size: ",len(dataset))
#     sample_x,sample_y = dataset[0]
#     print("sample shape:", sample_x.shape, "H =",sample_y.item())
    
    

    