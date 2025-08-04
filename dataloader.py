import torch
from torch.utils.data import DataLoader, Dataset
from data_simulator import simulate_fbm, add_noise_and_jumps
import numpy as np


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
            inc = simulate_fbm(H=H_true,T=1.0,n=1024)
            inc_noisy = add_noise_and_jumps(inc,sigma_micro=sigma_micro,jump_rate=jump_rate,jump_size = jump_size)
            data_list.append((inc_noisy,H_true))

    print(f'Building dataset with {len(data_list)} samples.')
    dataset = FBMDataset(data_list)

    return DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 2)



