from train import train_model
from evaluate import evaluate_model, predict_H
from dataloader import make_dataloader,make_dataloader_stock_path
from data_simulator import simulate_fbm_increments, add_noise_and_jumps,simulate_S,construct_cov
from model import HEstimatorCNN,HEstimatorFNN,estimate_H_mle_from_fBM,mle_for_inc,estimate_H_mle_from_stock,calc_spot_var
import torch
import numpy as np
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
import pickle


def compare_clean():
    #prep data
    H_vals = np.linspace(0.01,0.99,9)
    loader = make_dataloader(H_vals,n_paths = 1000,batch_size=32,sigma_micro=0.00,jump_rate=0.0,jump_size=0.00)

    #build test set
    N_test = 200
    Hs_test = np.random.uniform(0.01,0.99,N_test)
    incs_test,labels = [],[]
    for H in Hs_test:
        inc = simulate_fbm_increments(H,T=1.0,n=1024)
        inc = add_noise_and_jumps(inc,sigma_micro = 0.00,jump_rate=0.0,jump_size=0.00)
        incs_test.append(inc)
        labels.append(H)

    preds_mle = []
    dt= 1.0/1024
    preds_mle = Parallel(n_jobs=-1,verbose=10)(
        delayed(mle_for_inc)(inc, dt)
        for inc in incs_test
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
    plt.savefig('calibration_scatter_mle.png')
    print("Saved plot to calibration_scatter_mle.png")
    plt.close()

    # import SPX data
    SPX_var = np.loadtxt("SPX3yrDailyRealizedVariance1minBins.txt")
    SPX_log_var = np.log(SPX_var)
    SPX_fbm = SPX_log_var - SPX_log_var.mean()
    incs_SPX = np.diff(SPX_fbm)


    #train CNN
    device = torch.device("cpu")


    cnn = HEstimatorCNN()
    title_cnn = "cnn"
    cnn = train_model(cnn, loader, device, epochs=30, lr=1e-3, patience=5)
    bias_cnn,rmse_cnn = evaluate_model(cnn,device,incs_test,labels,title_cnn)
    print(f'CNN -> bias={bias_cnn:.4f}, RMSE={rmse_cnn:.4f}')
    H_cnn = predict_H(cnn,device,incs_SPX)
    print(f"CNN estimate for H of SPX data: {H_cnn}")


    #train FNN
    fnn = HEstimatorFNN()
    title_fnn = "fnn"
    fnn = train_model(fnn,loader,device,epochs=30,lr=1e-3,patience=5)
    bias_fnn,rmse_fnn = evaluate_model(fnn,device,incs_test,labels,title_fnn)
    print(f'FNN -> bias={bias_fnn:.4f}, RMSE={rmse_fnn:.4f}')
    # this doesnt work
    #H_fnn = predict_H(fnn,device,incs_SPX)
    #print(f'FNN estimate for H of SPX data: {H_fnn}')    


def compare_noise():
    #prep data
    H_vals = np.linspace(0.01,0.99,9)
    loader = make_dataloader(H_vals,n_paths = 1000,batch_size=32,sigma_micro=0.02,jump_rate=0.1,jump_size=0.01)

    #build test set
    N_test = 200
    Hs_test = np.random.uniform(0.01,0.99,N_test)
    incs_test,labels = [],[]
    for H in Hs_test:
        inc = simulate_fbm_increments(H,T=1.0,n=1024)
        inc = add_noise_and_jumps(inc,sigma_micro = 0.02,jump_rate=0.1,jump_size=0.01)
        incs_test.append(inc)
        labels.append(H)

    preds_mle = []
    dt= 1.0/1024
    preds_mle = Parallel(n_jobs=-1,verbose=10)(
        delayed(mle_for_inc)(inc, dt)
        for inc in incs_test
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
    plt.savefig('calibration_scatter_mle.png')
    print("Saved plot to calibration_scatter_mle.png")
    plt.close()

    # import SPX data
    SPX_var = np.loadtxt("SPX3yrDailyRealizedVariance1minBins.txt")
    SPX_log_var = np.log(SPX_var)
    SPX_fbm = SPX_log_var - SPX_log_var.mean()
    incs_SPX = np.diff(SPX_fbm)


    #train CNN
    device = torch.device("cpu")


    cnn = HEstimatorCNN()
    title_cnn = "cnn"
    cnn = train_model(cnn, loader, device, epochs=30, lr=1e-3, patience=5)
    bias_cnn,rmse_cnn = evaluate_model(cnn,device,incs_test,labels,title_cnn)
    print(f'CNN -> bias={bias_cnn:.4f}, RMSE={rmse_cnn:.4f}')
    H_cnn = predict_H(cnn,device,incs_SPX)
    print(f"CNN estimate for H of SPX data: {H_cnn}")


    #train FNN
    fnn = HEstimatorFNN()
    title_fnn = "fnn"
    fnn = train_model(fnn,loader,device,epochs=30,lr=1e-3,patience=5)
    bias_fnn,rmse_fnn = evaluate_model(fnn,device,incs_test,labels,title_fnn)
    print(f'FNN -> bias={bias_fnn:.4f}, RMSE={rmse_fnn:.4f}')
    # this doesnt work
    #H_fnn = predict_H(fnn,device,incs_SPX)
    #print(f'FNN estimate for H of SPX data: {H_fnn}')   

def simulate_fbm_from_stock_for_H(H, num_steps=10000, T=1.0):
    cov = construct_cov(num_steps=num_steps, H=H, T=T)
    L = np.linalg.cholesky(cov)
    stock_path = simulate_S(num_steps=num_steps, T=T, L=L)
    spot_var = calc_spot_var(stock_path,32)
    X = np.log(spot_var)
    fbm = X - X.mean()
    return fbm


def compare_stock_price(n_steps,n_paths,new_data):
    #prep data
    H_vals = np.linspace(0.01,0.99,9)
    loader = make_dataloader_stock_path(H_vals,n_paths = n_paths,n_steps=n_steps,batch_size=32,new_data=False)

    #build test set

    fbm_paths_test,labels = [],[]
    
    if new_data == True:
        N_test = 200
        Hs_test = np.random.uniform(0.01,0.99,N_test)
        fbm_paths_test = Parallel(n_jobs=-4, verbose=10)(
        delayed(simulate_fbm_from_stock_for_H)(H,n_steps) for H in Hs_test
        )
        with open("simulated_fbm_from_stock_data_test.pkl", "wb") as f:
            pickle.dump(fbm_paths_test, f)
            print("fbm from stock test data saved")
        with open("labels_for_fbm_from_stock_data_test.pkl","wb") as f:
            pickle.dump(Hs_test,f)
            print("labels for fbm from stock test data saved")
        labels = Hs_test

    else:
        with open("simulated_fbm_from_stock_data_test.pkl", "rb") as f:
            fbm_paths_test = pickle.load(f)
            print("fbm from stock test data loaded")
        with open("labels_for_fbm_from_stock_data_test.pkl","rb") as f:
            labels = pickle.load(f)
    


    preds_mle = []
    dt= 1.0/n_steps
    
    preds_mle = Parallel(n_jobs=-1,verbose=10)(
        delayed(estimate_H_mle_from_fBM)(fbm, dt)
        for fbm in fbm_paths_test
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
    cnn = train_model(cnn, loader, device, epochs=30, lr=1e-3, patience=5)
    bias_cnn,rmse_cnn = evaluate_model(cnn,device,fbm_paths_test,labels,title_cnn)
    print(f'CNN -> bias={bias_cnn:.4f}, RMSE={rmse_cnn:.4f}')

    # import SPX data
    SPX_var = np.loadtxt("SPX3yrDailyRealizedVariance1minBins.txt")
    SPX_log_var = np.log(SPX_var)
    SPX_fbm = SPX_log_var - SPX_log_var.mean()


    # Predict H for SPX data
    H_cnn = predict_H(cnn,device,SPX_fbm)
    print(f"CNN estimate for H of SPX data: {H_cnn}")


    #train FNN
    fnn = HEstimatorFNN(len(fbm_paths_test[0]))
    title_fnn = "fnn"
    fnn = train_model(fnn,loader,device,epochs=30,lr=1e-3,patience=5)
    bias_fnn,rmse_fnn = evaluate_model(fnn,device,fbm_paths_test,labels,title_fnn)
    print(f'FNN -> bias={bias_fnn:.4f}, RMSE={rmse_fnn:.4f}')
    # this doesnt work
    H_fnn = predict_H(fnn,device,incs_SPX)
    print(f'FNN estimate for H of SPX data: {H_fnn}')   
    
    



if __name__=="__main__":
     compare_stock_price(10240,1024,True)