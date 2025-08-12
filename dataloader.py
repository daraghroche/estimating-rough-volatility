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


class FBMDataset(Dataset):
    def __init__(self,data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        inc,H = self.data[idx]
        # convert to Torch tensors
        x = torch.from_numpy(inc.astype(np.float32))
        y = torch.tensor([H],dtype=torch.float32)
        return x,y

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
        new_data
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
                data_list.append((stock_path,H_true))



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
    dataset = FBMDataset(data_list)

    return DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 2)
