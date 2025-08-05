import numpy as np
import torch, os
from data_simulator import simulate_fbm_increments,add_noise_and_jumps
from model import HEstimatorCNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_model(model,device,incs_test,labels,title):
    model.eval()
    preds = []
    with torch.no_grad():
        for inc in incs_test:
            x = torch.from_numpy(inc.astype(np.float32)).to(device)
            x = x.unsqueeze(0)
            preds.append(model(x).item())
    errors=np.array(preds)-np.array(labels)
    plt.figure()
    plt.scatter(labels,preds,alpha=.7)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('True H')
    plt.ylabel('Predicted H')
    plt.title(f'Calibration of {title} estimator')
    plt.ylim(0, 1)
    plt.savefig(f'calibration_scatter_{title}.png')
    print("Saved plot to calibration_scatter.png")
    plt.close()
    return errors.mean(),np.sqrt((errors**2).mean())

def predict_H(model,device,incs):
    model.eval()
    x = torch.tensor(incs, dtype = torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).item()

