import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import make_dataloader
from model import HEstimatorCNN
import numpy as np

def train_model(model, loader, device, epochs=30, lr=1e-3, patience=5,name = "cnn"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(),lr=lr)
    sched = optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    #crit = nn.MSELoss()
    crit = nn.SmoothL1Loss(beta = 0.05)
    best_loss = float("inf")
    no_improve = 0



    for epoch in range(1,epochs+1):
        model.train()
 
        total = 0
        


        for x,y in loader:
            
            
                
            x,y = x.to(device),y.to(device)
                
            opt.zero_grad(set_to_none=True)
                
            pred=model(x)                
            loss = crit(pred,y)
         
            loss.backward()
                
            opt.step()
            total += loss.item() * x.size(0)
            train_mse = total/len(loader.dataset)
                
        print(f'Epoch {epoch:2d}:MSE={train_mse:.3e} lr={sched.get_last_lr()[0]:.1e}')

        if train_mse + 1e-6 <best_loss:
            best_loss=train_mse
            no_improve = 0
        else:
            no_improve +=1
            if no_improve>=patience:
                print(f'stopping early (no improvement in {patience} epochs)')        
                break
        sched.step()
    return model


def main():
    H_values = np.linspace(0.1,.9,10)
    loader = make_dataloader(H_values,n_paths = 1000,batch_size=32,sigma_micro=0.0,jump_size=0.0,jump_rate=0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HEstimatorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(1,31):
        model.train()
        total = 0
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=loss.item()*x.size(0)
        print(f'Epoch {epoch},MSE={total/len(loader.dataset):.3e}')
    torch.save(model.state_dict(),"model_checkpoint.pt")
    print("model saved to module_checkpoint.pt")

# if __name__ == "__main__":
#     main()