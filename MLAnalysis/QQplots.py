import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from utils.ice import generate_ice_data, plot_ice, load_model
#import scipy.stats as stats
from utils.utils import OFLESDataset, R2Score, DataCollecter, MOTHERDIR, M2_HEADERS, M5_HEADERS

from model.wae import WAE
from model.mlp import mlp
from utils.loss import WaveletLoss

modelMode = 'WAE' # 'MLP' #
train_org, train_norm, train_means, train_scales = DataCollecter()
dt = train_norm.filter(globals()[f"M5_HEADERS"], axis=1)
dt_names = ['M5']
org, norm, means, scales = DataCollecter()
ncount = 21
test_org, test_norm, test_means, test_scales = org[::ncount], norm[::ncount], means[::ncount], scales[::ncount]
fontSize = 18
Num = 45000
for dt_name in dt_names:
    print(f'Model Config {dt_name}:')
    dt = test_norm.sample(n=Num, random_state=42).reset_index(drop=True)
    dt = dt.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
    ds = OFLESDataset(dt)
    ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=Num, shuffle=False)

    feature_names = globals()[f"{dt_name}_HEADERS"]
    device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
    model_path = f'{MOTHERDIR}/best_model.pt'
    model = load_model(modelMode, model_path, ds[0].shape[0] - 1, 1, device)

    for batch_idx, batch in enumerate(ds_loader):
        print(f'Processing batch {batch_idx+1}/{len(ds_loader)}')
        features = batch[:, 0:-1].to(device)
        target = batch[:, -1].to(device)

        # Model output
        model_output = model(features).detach().cpu().numpy().squeeze()

        sorted_ground_truth = np.sort(target)
        sorted_predictions = np.sort(model_output)

        # Create the Q-Q plot
        plt.figure(figsize=(8,7))
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
        plt.plot(sorted_ground_truth, sorted_predictions, 'o', markersize=4, markeredgewidth=1, markeredgecolor='blue', markerfacecolor='none')
        plt.plot([sorted_ground_truth.min(), sorted_ground_truth.max()], [sorted_ground_truth.min(), sorted_ground_truth.max()], 'r--')  # 45-degree line
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.xlabel(r'$\nu^{\hbox{\tiny SA}}_\tau$', fontsize=fontSize)
        plt.ylabel(r'$\nu^{\hbox{\tiny SAM}}_\tau$', fontsize=fontSize)
        plt.xlim([-2,10])
        plt.ylim([-2,10])
        plt.xticks(fontsize=fontSize)  
        plt.yticks(fontsize=fontSize)
        ax = plt.gca()
        ax.xaxis.get_offset_text().set_fontsize(fontSize)  
        ax.yaxis.get_offset_text().set_fontsize(fontSize)
        #plt.title('Q-Q Plot: Ground Truth vs. Predictions')
        #plt.grid(True)
        plt.savefig('ModelOutput_QQ_twoVars.png')
        plt.close()
        
        """
        plt.figure()
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
        stats.probplot(model_output, dist="norm", plot=plt)
        plt.xlim([-4,4])
        plt.ylim([-8,8])
        plt.title('')
        plt.savefig('./ModelOutput_QQ.png')
        plt.close()
        """
        break
