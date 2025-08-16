import torch
from torch.utils.data import DataLoader, Dataset
from data_simulator import simulate_fbm_increments, add_noise_and_jumps,simulate_S,construct_cov
import numpy as np
from joblib import Parallel, delayed
from model import calc_spot_var
import pickle
import gc
import scipy.linalg as la
from scipy.linalg import blas
import random


class FBMDataset(Dataset):
    def __init__(self,data_list,crop_len = 1024, mode = "random",crops_per_path = 1,pad_mode = "reflect"):
        #self.data = data_list
        self.paths = [np.asarray(p[0],dtype = np.float32) for p in data_list]
        self.labels = np.asarray([p[1] for p in data_list], dtype = np.float32)
        self.crop_len = int(crop_len)
        self.mode = mode
        self.crops_per_pat = int(max(1,crops_per_path))
        self.pad_mode = pad_mode

        for i,x in enumerate(self.paths):
            if x.ndim != 1:
                raise ValueError(f'path {i} must be 1D got {x.shape}')
            if x.shape[0]<self.crop_len:
                if self.pad_mode is None:
                    raise ValueError(f'path {i} is shorter than crop_len = {self.crop_len}')
                pad = self.crop_len-x.shape[0]
                self.paths[i]= np.pad(x, (0,pad),mode = self.pad_mode)
    
    def __len__(self):
        return len(self.paths)*(self.crops_per_pat if self.mode == "random" else 1)
    
    def __getitem__(self,idx):
        base_idx = idx if self.mode !="random" else idx//self.crops_per_pat
        x = self.paths[base_idx]
        y = self.labels[base_idx]
        T = x.shape[0]
        if self.mode == "random":
            start = random.randint(0,T-self.crop_len)
        elif self.mode == "center":
            start = max(0,(T-self.crop_len)//2)
        else:
            start = 0

        crop = x[start:start + self.crop_len]
        xt = torch.from_numpy(crop).float()
        yt = torch.tensor([y],dtype=torch.float32)
        #inc,H = self.data[idx]
        return xt,yt
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() %2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataloader(
        H_values,
        n_paths,
        batch_size,
        sigma_micro,
        jump_size,
        jump_rate
):
    data_list = []
    for H_true in H_values:
        for _ in range(n_paths):
            inc = simulate_fbm_increments(H=H_true,T=1.0,n=1024)
            inc_noisy = add_noise_and_jumps(inc,sigma_micro=sigma_micro,jump_rate=jump_rate,jump_size = jump_size)
            data_list.append((inc_noisy,H_true))

    print(f'Building dataset with {len(data_list)} samples.')
    dataset = FBMDataset(data_list)

    return DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 2)


def make_dataloader_stock_path(
        H_values,
        n_paths,
        n_steps,
        batch_size,
        new_data,
        crop_len = 1024,
        crops_per_path= 4,
        mode = "random",
        num_workers = 2
):
    data_list = []
    H_count = 0
    if new_data == True:
        for H_true in H_values:
            print("start cov")
            cov = construct_cov(num_steps = n_steps,H=H_true,T=1.0)
            print("start chol")
            L = np.linalg.cholesky(cov)
            #L = la.cholesky(cov, lower=True, overwrite_a=True, check_finite=False).astype(np.float32, copy=False)
 
            Zs = np.random.randn(n_paths, n_steps-1)
            print("starting BHs calc")
            BHs = (L@Zs.T).T
            #BHs = (blas.dtrmm(alpha = 1.0,a=L,b=Zs.T,lower=True)).T
            print("start sims")
            for _ in range(n_paths):
                stock_path = simulate_S(num_steps=n_steps,T=1.0,nu=1,S0=1,p=-.65,V0=.1,BH=BHs[_],Z=Zs[_])
                ratio = stock_path[1:] / stock_path[:-1]
                log_ret = np.log(ratio, where=(ratio>0), out=np.full_like(ratio, np.nan))
                log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)
                data_list.append((log_ret, H_true))




            H_count+=1
            print(f'H count: {H_count}')
        with open("simulated_stock_data_training.pkl", "wb") as f:
            pickle.dump(data_list, f)
            print("stock training data saved")
    else:
        with open("simulated_stock_data_training.pkl", "rb") as f:
            data_list = pickle.load(f)
            print("stock training data loaded")
    print(f'Building dataset with {len(data_list)} samples.')
    dataset = FBMDataset(data_list,crop_len = crop_len,mode=mode,crops_per_path = crops_per_path)

    return DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = num_workers,worker_init_fn= seed_worker)
