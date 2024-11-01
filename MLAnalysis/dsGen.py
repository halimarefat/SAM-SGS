import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


def dataReader(path):
    if path.split('/')[-1] == 'UMean':
        f = open(path, 'r')
        for _ in range(20):
            f.readline()
        count = int(f.readline())
        f.readline()
        vec = np.zeros((count,3))
        for i in range(count):
            tmp = f.readline().replace('(', ' ').replace(')', ' ').split(' ')
            vec[i, 0] = tmp[1]
            vec[i, 1] = tmp[2]
            vec[i, 2] = tmp[3]
        print("The {} is done!".format(path.split('/')[-1]))
        return vec
    elif path.split('/')[-1] == 'UPrime2Mean':
        f = open(path, 'r')
        for _ in range(20):
            f.readline()
        count = int(f.readline())
        f.readline()
        vec = np.zeros((count,6))
        for i in range(count):
            tmp = f.readline().replace('(', ' ').replace(')', ' ').split(' ')
            vec[i, 0] = tmp[1]
            vec[i, 1] = tmp[2]
            vec[i, 2] = tmp[3]
            vec[i, 3] = tmp[4]
            vec[i, 4] = tmp[5]
            vec[i, 5] = tmp[6]
        print("The {} is done!".format(path.split('/')[-1]))
        return vec
    else:
        f = open(path, 'r')
        for _ in range(20):
            f.readline()
        count = int(f.readline())
        f.readline()
        q = []
        for _ in range(count):
            q.append(np.float64(f.readline()))
        print("The {} is done!".format(path.split('/')[-1]))
        return q
    
    

def dataCollector(main_path):
    _UP2Mean = dataReader(main_path+'UPrime2Mean')
    _UMean = dataReader(main_path+'UMean')
    _enstrophy = dataReader(main_path+'enstrophy') 
    _epsilon = dataReader(main_path+'epsilon')
    _k_sgs = dataReader(main_path+'k')
    _Qg = dataReader(main_path+'Qg')
    _Rg = dataReader(main_path+'Rg')
    _Qs = dataReader(main_path+'Qs')
    _Rs = dataReader(main_path+'Rs')
    _skew = dataReader(main_path+'skewness')
    _vortStr = dataReader(main_path+'vortexStretching') 
    
    _UMeanX = []
    _UMeanY = []
    _UMeanZ = []
    for i in range(len(_UMean)):
        _UMeanX.append(_UMean[i, 0])
        _UMeanY.append(_UMean[i, 1])
        _UMeanZ.append(_UMean[i, 2])
    
    _UP2MeanXX = []
    _UP2MeanYY = []
    _UP2MeanZZ = []
    for i in range(len(_UMean)):
        _UP2MeanXX.append(_UP2Mean[i, 0])
        _UP2MeanYY.append(_UP2Mean[i, 3])
        _UP2MeanZZ.append(_UP2Mean[i, 5])
    
    data = {'enstrophy':_enstrophy, 'epsilon':_epsilon, 'k_sgs':_k_sgs, 'Qg':_Qg, 'Rg':_Rg, 'Qs':_Qs, 'Rs':_Rs, 
        'skewness':_skew, 'vortex-stretching':_vortStr, 
        'UMeanX': _UMeanX, 'UMeanY': _UMeanY, 'UMeanZ': _UMeanZ, 
        'UP2MXX': _UP2MeanXX, 'UP2MYY': _UP2MeanYY, 'UP2MZZ': _UP2MeanZZ}
    ds = pd.DataFrame(data=data)

    print('\ndata is collected and ds is ready!\n')
    
    return ds


    
SA_path = '/path/to/SA_Case/Timestep/'
SA_ds = dataCollector(main_path=SA_path)
SA_ds.to_csv('SA_ds.csv')
print('\nSA is done!\n\n')

SAM_path = '/path/to/SAM_Case/Timestep/'
SAM_ds = dataCollector(main_path=SAM_path)
SAM_ds.to_csv('SAM_ds.csv')
print('\nSAM is done!\n\n')

