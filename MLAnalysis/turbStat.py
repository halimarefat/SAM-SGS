import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.integrate import trapz
from scipy.stats import gaussian_kde

SA_ds = pd.read_csv('SA_ds.csv')
SAM_ds = pd.read_csv('SAM_ds.csv')

ss = 1
ee = 99
fontSize = 23

n, xedges, yedges = np.histogram2d(SA_ds['Rg'], SA_ds['Qg'], bins=[1500, 1501])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

jpdf_min = np.min(jpdf)
jpdf_max = np.max(jpdf)

# Print to inspect
print(f"Min value in jpdf: {jpdf_min}, Max value in jpdf: {jpdf_max}")

lower_bound = np.percentile(jpdf, ss)  # 1st percentile
upper_bound = np.percentile(jpdf, ee)  # 99th percentile

plt.figure(figsize=(10,7))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.pcolormesh(X, Y, jpdf.T, shading='auto') 
#plt.clim([-1.5e-5,3.5e-4])
plt.clim([lower_bound, upper_bound])
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fontSize)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontSize)
plt.xlabel(r'$\mathcal{R}_{g}/\langle \omega^2\rangle^{3/2}$', fontsize=fontSize)
plt.ylabel(r'$\mathcal{Q}_{g}/ \omega^2 $', fontsize=fontSize)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xticks(fontsize=fontSize)  
plt.yticks(fontsize=fontSize)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(fontSize)  
ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.xlim([-0.35,0.8])
#plt.ylim([-1e4,1e4])
plt.savefig('./Results/RgQg_SA.png')
plt.close()



n, xedges, yedges = np.histogram2d(SAM_ds['Rg'], SAM_ds['Qg'], bins=[1500, 1501])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

jpdf_min = np.min(jpdf)
jpdf_max = np.max(jpdf)

# Print to inspect
print(f"Min value in jpdf: {jpdf_min}, Max value in jpdf: {jpdf_max}")

lower_bound = np.percentile(jpdf, ss)  # 1st percentile
upper_bound = np.percentile(jpdf, ee)  # 99th percentile

plt.figure(figsize=(10,7))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.pcolormesh(X, Y, jpdf.T, shading='auto') 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fontSize)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.clim([-1.5e-5,3.5e-4])
plt.clim([lower_bound, upper_bound])
plt.xlabel(r'$\mathcal{R}_{g}/\langle \omega^2\rangle^{3/2}$', fontsize=fontSize)
plt.ylabel(r'$\mathcal{Q}_{g}/ \omega^2 $', fontsize=fontSize)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xticks(fontsize=fontSize)  
plt.yticks(fontsize=fontSize)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(fontSize)  
ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.xlim([-0.35,0.8])
#plt.ylim([-1e4,1e4])
plt.savefig('./Results/RgQg_SAM.png')
plt.close()


n, xedges, yedges = np.histogram2d(SA_ds['Rs'], SA_ds['Qs'], bins=[2000, 2001])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

jpdf_min = np.min(jpdf)
jpdf_max = np.max(jpdf)

# Print to inspect
print(f"Min value in jpdf: {jpdf_min}, Max value in jpdf: {jpdf_max}")

lower_bound = np.percentile(jpdf, ss)  # 1st percentile
upper_bound = np.percentile(jpdf, ee)  # 99th percentile

plt.figure(figsize=(10,7))
plt.pcolormesh(X, Y, jpdf.T, shading='auto') 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fontSize)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.clim([6.99811*1e-10,6.99822*1e-10])
plt.clim([lower_bound, upper_bound])
plt.xlabel(r'$\mathcal{R}_{s}/\langle \omega^2\rangle^{3/2}$', fontsize=fontSize)
plt.ylabel(r'$\mathcal{Q}_{s}/ \omega^2 $', fontsize=fontSize)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xticks(fontsize=fontSize)  
plt.yticks(fontsize=fontSize)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(fontSize)  
ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.xlim([-70,70])
#plt.ylim([-60,0])
plt.savefig('./Results/RsQs_SA.png')

n, xedges, yedges = np.histogram2d(SAM_ds['Rs'], SAM_ds['Qs'], bins=[2000, 2001])
jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

jpdf_min = np.min(jpdf)
jpdf_max = np.max(jpdf)

# Print to inspect
print(f"Min value in jpdf: {jpdf_min}, Max value in jpdf: {jpdf_max}")

lower_bound = np.percentile(jpdf, ss)  # 1st percentile
upper_bound = np.percentile(jpdf, ee)  # 99th percentile

plt.figure(figsize=(10,7))
plt.pcolormesh(X, Y, jpdf.T, shading='auto') 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fontSize)
cbar.ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.clim([6.99811*1e-10,6.99822*1e-10])
plt.clim([lower_bound, upper_bound])
plt.xlabel(r'$\mathcal{R}_{s}/\langle \omega^2\rangle^{3/2}$', fontsize=fontSize)
plt.ylabel(r'$\mathcal{Q}_{s}/ \omega^2 $', fontsize=fontSize)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xticks(fontsize=fontSize)  
plt.yticks(fontsize=fontSize)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(fontSize)  
ax.yaxis.get_offset_text().set_fontsize(fontSize)
#plt.xlim([-70,70])
#plt.ylim([-60,0])
plt.savefig('./Results/RsQs_SAM.png')
