import numpy as np
import torch
import time
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from tqdm import tqdm
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import OFLESDataset, R2Score, DataCollecter, MOTHERDIR, M2_HEADERS, M5_HEADERS

from model.wae import WAE

modelMode = 'SAM-SGS'#'WAE' # 'MLP' #
train_org, train_norm, train_means, train_scales = DataCollecter()
dt = train_norm.filter(globals()[f"M5_HEADERS"], axis=1)
dt_names = ['M5']
org, norm, means, scales = DataCollecter()
ncount = 50
test_org, test_norm, test_means, test_scales = org[::ncount], norm[::ncount], means[::ncount], scales[::ncount]


fontSize = 18

Num = 45000
for dt_name in dt_names:
    print(f'Working on {dt_name}!')
    #dt = test_norm.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
    dt = test_norm.sample(n=Num, random_state=42).reset_index(drop=True)
    dt = dt.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
    ds = OFLESDataset(dt)
    test_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=50000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PATH = f"{MOTHERDIR}/best_model.pt"
    out_channels = 1
    in_channels = dt.shape[1] - out_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)    
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    model.double()
    criterion = nn.MSELoss()

    nut_true = []
    nut_pred = []
        
    start_time = time.time()
    test_loss = 0.0
    test_loop = tqdm(test_loader, position=0, leave=True)
    with torch.no_grad():
        for batch in test_loop:
            features = batch[:, :-1].to(device)
            label = batch[:, -1].to(device)
            output = model(features)
            pred = output.squeeze()
            loss = criterion(pred, label)
            test_loss += loss.item()
            test_loop.set_postfix(loss=loss.item())
            nut_pred.append(pred.detach().cpu().numpy() * test_scales['Nut'].values + test_means['Nut'].values)
            nut_true.append(label.detach().cpu().numpy() * test_scales['Nut'].values + test_means['Nut'].values)

    nut_true = np.concatenate(nut_true).ravel()
    nut_pred = np.concatenate(nut_pred).ravel()
    test_tau = (nut_true * dt['S1'] + # \tau_11 
                nut_true * dt['S2'] + # \tau_12
                nut_true * dt['S3'] + # \tau_13
                nut_true * dt['S2'] + # \tau_21
                nut_true * dt['S4'] + # \tau_22
                nut_true * dt['S5'] + # \tau_23
                nut_true * dt['S3'] + # \tau_31
                nut_true * dt['S5'] + # \tau_32
                nut_true * dt['S6'])  # \tau_33

    test_tau_S = (nut_true * dt['S1'] * dt['S1'] + # \tau_11 
                  nut_true * dt['S2'] * dt['S2'] + # \tau_12
                  nut_true * dt['S3'] * dt['S3'] + # \tau_13
                  nut_true * dt['S2'] * dt['S2'] + # \tau_21
                  nut_true * dt['S4'] * dt['S4'] + # \tau_22
                  nut_true * dt['S5'] * dt['S5'] + # \tau_23
                  nut_true * dt['S3'] * dt['S3'] + # \tau_31
                  nut_true * dt['S5'] * dt['S5'] + # \tau_32
                  nut_true * dt['S6'] * dt['S6'])  # \tau_33

    test_tau_M = (nut_pred * dt['S1'] + # \tau_11 
                  nut_pred * dt['S2'] + # \tau_12
                  nut_pred * dt['S3'] + # \tau_13
                  nut_pred * dt['S2'] + # \tau_21
                  nut_pred * dt['S4'] + # \tau_22
                  nut_pred * dt['S5'] + # \tau_23
                  nut_pred * dt['S3'] + # \tau_31
                  nut_pred * dt['S5'] + # \tau_32
                  nut_pred * dt['S6'])  # \tau_33
    
    test_tau_S_M = (nut_pred * dt['S1'] * dt['S1'] + # \tau_11 
                    nut_pred * dt['S2'] * dt['S2'] + # \tau_12
                    nut_pred * dt['S3'] * dt['S3'] + # \tau_13
                    nut_pred * dt['S2'] * dt['S2'] + # \tau_21
                    nut_pred * dt['S4'] * dt['S4'] + # \tau_22
                    nut_pred * dt['S5'] * dt['S5'] + # \tau_23
                    nut_pred * dt['S3'] * dt['S3'] + # \tau_31
                    nut_pred * dt['S5'] * dt['S5'] + # \tau_32
                    nut_pred * dt['S6'] * dt['S6'])  # \tau_33
    
    


    sorted_ground_truth = np.sort(test_tau)
    sorted_predictions = np.sort(test_tau_M)
    # Create the Q-Q plot
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })
    plt.plot(sorted_ground_truth, sorted_predictions, 'o', markersize=4, markeredgewidth=1, markeredgecolor='blue', markerfacecolor='none')
    plt.plot([sorted_ground_truth.min(), sorted_ground_truth.max()], [sorted_ground_truth.min(), sorted_ground_truth.max()], 'r--')  # 45-degree line
    plt.xlabel(r'$\tau^{\hbox{\tiny SA}}_{ij}$', fontsize=fontSize)
    plt.ylabel(r'$\tau^{\hbox{\tiny SAM}}_{ij}$', fontsize=fontSize)
    #plt.xlim([-2,10])
    #plt.ylim([-2,10])
    #plt.title('Q-Q Plot: Ground Truth vs. Predictions')
    #plt.grid(True)
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.savefig('ModelOutput_QQ_twoVars_tau.png')
    plt.close()
    
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.scatter(test_tau, test_tau_M, color='red', edgecolor='white')
    plt.plot([-0.003,0.003], [-0.003,0.003], 'b--')
    plt.xlabel(r'$\tau^{\hbox{\tiny SA}}_{ij}$', fontsize=fontSize)
    plt.ylabel(r'$\tau^{\hbox{\tiny SAM}}_{ij}$', fontsize=fontSize)
    #plt.xlim([-1,0.0004])
    #plt.ylim([-1,0.0004])
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.savefig(f'{dt_name}_scatter_tau.png')
    plt.close()
    
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.hist(test_tau, bins=1000, density=True, alpha=0.6, histtype=u'step', color='blue')
    plt.hist(test_tau_M, bins=1000, density=True, alpha=0.6, histtype=u'step', color='red')
    plt.xlim([-0.001, 0.001])
    #plt.ylim([0, 85])
    plt.xlabel(r'$\tau^M_{ij}$', fontsize=fontSize)
    plt.ylabel(r'density', fontsize=fontSize)
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.legend(['SA-SGS', modelMode], frameon=False)
    plt.savefig(f'{dt_name}_density_tau.png')
    plt.close()
    
    
    sorted_ground_truth = np.sort(test_tau_S)
    sorted_predictions = np.sort(test_tau_S_M)
    # Create the Q-Q plot
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })
    plt.plot(sorted_ground_truth, sorted_predictions, 'o', markersize=4, markeredgewidth=1, markeredgecolor='blue', markerfacecolor='none')
    plt.plot([sorted_ground_truth.min(), sorted_ground_truth.max()], [sorted_ground_truth.min(), sorted_ground_truth.max()], 'r--')  # 45-degree line
    plt.xlabel(r'$\tau^{\hbox{\tiny SA}}_{ij}\mathcal{S}_{ij}$', fontsize=fontSize)
    plt.ylabel(r'$\tau^{\hbox{\tiny SAM}}_{ij}\mathcal{S}_{ij}$', fontsize=fontSize)
    #plt.xlim([-2,10])
    #plt.ylim([-2,10])
    #plt.title('Q-Q Plot: Ground Truth vs. Predictions')
    #plt.grid(True)
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.savefig('ModelOutput_QQ_twoVars_tau_S.png')
    plt.close()
    
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.scatter(test_tau_S, test_tau_S_M, color='red', edgecolor='white')
    plt.plot([0,0.014], [0,0.014], 'b--')
    plt.xlabel(r'$\tau^{\hbox{\tiny SA}}_{ij}\mathcal{S}_{ij}$', fontsize=fontSize)
    plt.ylabel(r'$\tau^{\hbox{\tiny SAM}}_{ij}\mathcal{S}_{ij}$', fontsize=fontSize)
    #plt.xlim([-1,0.0004])
    #plt.ylim([-1,0.0004])
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.savefig(f'{dt_name}_scatter_tau_S.png')
    plt.close()
    
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.hist(test_tau_S, bins=1000, density=True, alpha=0.6, histtype=u'step', color='blue')
    plt.hist(test_tau_S_M, bins=1000, density=True, alpha=0.6, histtype=u'step', color='red')
    plt.xlim([-0.0001, 0.0025])
    #plt.ylim([0, 85])
    plt.xlabel(r'$\tau_{ij}\mathcal{S}_{ij}$', fontsize=fontSize)
    plt.ylabel(r'$density$', fontsize=fontSize)
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    plt.legend(['SA-SGS', modelMode], frameon=False)
    plt.savefig(f'{dt_name}_density_tau_S.png')
    plt.close()
    