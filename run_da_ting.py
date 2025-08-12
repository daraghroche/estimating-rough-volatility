from train import train_model
from evaluate import evaluate_model, predict_H
from dataloader import make_dataloader,make_dataloader_stock_path
from data_simulator import simulate_fbm_increments, add_noise_and_jumps,simulate_S,construct_cov
from model import HEstimatorCNN,HEstimatorFNN,calc_spot_var,mle_from_stock,HurstNet
import torch
import numpy as np
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
import pickle
import gc
  

def simulate_fbm_from_stock_for_H(H, num_steps=10000, T=1.0):
    cov = construct_cov(num_steps=num_steps, H=H, T=T)
    L = np.linalg.cholesky(cov)

    stock_path = simulate_S(num_steps=num_steps, T=T, L=L)
    spot_var = calc_spot_var(stock_path,32)
    X = np.log(spot_var)
    fbm = X - X.mean()
    return fbm

def simulate_S_from_H(num_steps,T,H,nu,S0,p,V0):
    cov = construct_cov(num_steps,H,T)
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(num_steps-1)
    BH = L@Z

    stock = simulate_S(num_steps,T,nu,S0,p,V0,BH,Z)
    return stock


def compare_stock_price(n_steps,n_paths,new_data):
    #prep data
    H_vals = np.linspace(0.05,0.95,64)
    loader = make_dataloader_stock_path(H_vals,n_paths = n_paths,n_steps=n_steps,batch_size=32,new_data=new_data)

    #build test set

    stock_paths_test,labels = [],[]
        
    if new_data == True:
        N_test = 200
        Hs_test = np.random.uniform(0.01,0.99,N_test)
        stock_paths_test = Parallel(n_jobs=-2, verbose=10)(
        delayed(simulate_S_from_H)(num_steps = n_steps,T=1,H = H,nu = 1,S0 = 1,p=-.65,V0=.1) for H in Hs_test )
        with open("simulated_stock_data_test.pkl", "wb") as f:
            pickle.dump(stock_paths_test, f)
            print("stock test data saved")
        with open("labels_for_stock_data_test.pkl","wb") as f:
            pickle.dump(Hs_test,f)
            print("labels for stock test data saved")
        labels = Hs_test

    else:
        with open("simulated_stock_data_test.pkl", "rb") as f:
            stock_paths_test = pickle.load(f)
            print("stock test data loaded")
        with open("labels_for_stock_data_test.pkl","rb") as f:
            labels = pickle.load(f)
    


    preds_mle = []
    dt= 1.0/n_steps
    
    preds_mle = Parallel(n_jobs=-1,verbose=10)(
        delayed(mle_from_stock)(S = stock,T = 1,grad = 64)
        for stock in stock_paths_test
        )


    preds_mle = np.array(preds_mle)
    labels_mle = np.array(labels)

    bias_mle = np.mean(preds_mle - labels_mle)
    rmse_mle = np.sqrt(np.mean((preds_mle - labels)**2))
    print(f"MLE   → bias={bias_mle:.4f}, RMSE={rmse_mle:.4f}")

    plt.figure()
    plt.scatter(labels, preds_mle, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True H')
    plt.ylabel('Predicted H')
    plt.title('Calibration of MLE estimator')
    plt.ylim(0, 1)
    plt.savefig('calibration_scatter_mle.png')
    print("Saved plot to calibration_scatter_mle.png")
    plt.close()

    #train CNN
    device = torch.device("cpu")


    cnn = HEstimatorCNN()
    title_cnn = "cnn"
    cnn = train_model(cnn, loader, device, epochs=20, lr=1e-3, patience=5,name = "cnn")
    bias_cnn,rmse_cnn = evaluate_model(cnn,device,stock_paths_test,labels,title_cnn)
    print(f'CNN -> bias={bias_cnn:.4f}, RMSE={rmse_cnn:.4f}')



if __name__=="__main__":
     compare_stock_price(7000,4096,True)