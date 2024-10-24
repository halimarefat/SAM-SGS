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
from utils.utils import OFLESDataset, R2Score, DataCollecter, MOTHERDIR, M5_HEADERS, M2_HEADERS
from model.wae import WAE
from model.mlp import mlp
from utils.loss import WaveletLoss
from utils.sensitivity import sensitivity_analysis, plot_sensitivities, bland_altman_plot

modelMode = 'SAM-SGS'#'WAE' # 'MLP' #
train_org, train_norm, train_means, train_scales = DataCollecter()
dt = train_norm.filter(globals()[f"M5_HEADERS"], axis=1)
dt_names = ['M5']
test_org, test_norm, test_means, test_scales = DataCollecter()

for dt_name in dt_names:
    print(f'Working on {dt_name}!')
    out_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dt_1 = test_norm.filter(globals()[f"M5_HEADERS"], axis=1)
    ds_1 = OFLESDataset(dt_1)
    test_loader_1 = torch.utils.data.DataLoader(dataset=ds_1, batch_size=50000, shuffle=False)
    PATH_1 = "/home/hmarefat/scratch/Paper_3/WF_ADM_mesh5/_MLAnalysis/best_model.pt"
    in_channels_1 = dt_1.shape[1] - out_channels
    model_1 = WAE(in_channels=in_channels_1, out_channels=out_channels, bilinear=True) 
    model_1.load_state_dict(torch.load(PATH_1))
    model_1.eval()
    model_1.to(device)
    model_1 = model_1.float()
    
    
    criterion = nn.MSELoss()

    test_loop_1 = tqdm(test_loader_1, position=0, leave=True)

    
    Nut_true_1, Nut_pred_1 = [], []
    
    for batch_1 in test_loop_1:
        features_1 = batch_1[:, :-1].to(device).float()
        label_1 = batch_1[:, -1].to(device).float()
        output_1 = model_1(features_1)
        pred_1 = output_1.squeeze()
        Nut_pred_1.append(pred_1.detach().cpu().numpy() * test_scales['Nut'].values + test_means['Nut'].values)
        Nut_true_1.append(label_1.detach().cpu().numpy() * test_scales['Nut'].values + test_means['Nut'].values)

        sensitivities_1 = sensitivity_analysis(model_1, features_1)
        
        feature_names_1 = globals()[f"M5_HEADERS"][:-1]  
    
        colors = ['magenta'] 
        labels = dt_name
        #plot_sensitivities([feature_names_1, feature_names_2], [sensitivities_1, sensitivities_2], colors, labels, Path(f'{MOTHERDIR}/Results/{dt_name}_sensitivity.png'))
        break  
    
    Nut_true_1 = np.concatenate(Nut_true_1).ravel()
    Nut_pred_1 = np.concatenate(Nut_pred_1).ravel()

 
    bland_altman_plot(Nut_true_1, Nut_pred_1, Path(f'{MOTHERDIR}/Results/{dt_name[0]}_bland_altman.png'))
