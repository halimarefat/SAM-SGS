import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def TKEmetric(k_sgs, UP2MX, UP2MY, UP2MZ):
    k_res = 0.5 * (UP2MX + UP2MY + UP2MZ)
    metric = k_sgs / (k_sgs+k_res)
    mean = np.mean(metric)
    median = np.median(metric)
    std = np.std(metric)
    min_metric = np.min(metric)
    max_metric = np.max(metric)

    return metric, mean, median, std, min_metric, max_metric

def TKEmetricHist(metric, outFile):
    plt.figure(figsize=(10,7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.hist(metric, bins=260, edgecolor='blue')
    plt.xlabel(r'$k_{\hbox{\tiny{sgs}}} / (k_{\hbox{\tiny{sgs}}} + k_{\hbox{\tiny{res}}})$', fontsize=fontSize)
    plt.ylabel('Probability Density', fontsize=fontSize)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.ylim([0, 6e5])
    plt.xticks(fontsize=fontSize)  
    plt.yticks(fontsize=fontSize)
    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(fontSize)  
    ax.yaxis.get_offset_text().set_fontsize(fontSize)
    plt.savefig(outFile)
    plt.close()
    
SA_ds = pd.read_csv('SA_ds.csv')
SAM_ds = pd.read_csv('SAM_ds.csv')

fontSize = 21
SA_metric, SA_metric_mean, SA_metric_median, SA_metric_std, SA_metric_min, SA_metric_max = TKEmetric(SA_ds['k_sgs'], SA_ds['UP2MXX'], SA_ds['UP2MYY'], SA_ds['UP2MZZ'])
SAM_metric, SAM_metric_mean, SAM_metric_median, SAM_metric_std, SAM_metric_min, SAM_metric_max = TKEmetric(SAM_ds['k_sgs'], SAM_ds['UP2MXX'], SAM_ds['UP2MYY'], SAM_ds['UP2MZZ'])

print(f'Mean TKE metric for SA is {SA_metric_mean}, for SAM is {SAM_metric_mean}.')
print(f'Median TKE metric for SA is {SA_metric_median}, for SAM is {SAM_metric_median}.')
print(f'STD TKE metric for SA is {SA_metric_std}, for SAM is {SAM_metric_std}.')
print(f'Min TKE metric for SA is {SA_metric_min}, for SAM is {SAM_metric_min}.')
print(f'Max TKE metric for SA is {SA_metric_max}, for SAM is {SAM_metric_max}.')


TKEmetricHist(SA_metric, 'SA_metric_hist.png')
TKEmetricHist(SAM_metric, 'SAM_metric_hist.png')


