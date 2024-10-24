import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#df_path = '/home/hmarefat/scratch/Paper_3/WF_ADM_mesh5/postProcessing/fieldData_.dat'
#df = pd.read_csv(df_path)

#df_reduced = df.iloc[::25]  

output_path = './fieldData_noheader.csv'
#df_reduced.to_csv(output_path, index=False)

cols = ["t", "x", "y", "z"  
             , "Ux"   , "Uy"   , "Uz"
             , "G1"   , "G2"   , "G3"   , "G4"   , "G5"   , "G6"
             , "S1"   , "S2"   , "S3"   , "S4"   , "S5"   , "S6"  
             , "UUp1" , "UUp2" , "UUp3" , "UUp4" , "UUp5" , "UUp6" 
             , "nut"]
df = pd.read_csv(output_path, sep='\t', names=cols)

print(df.head())

ds_scaler = StandardScaler()
ds_scaler.fit(df)
np.savetxt('./fieldData_means.txt',ds_scaler.mean_)
np.savetxt('./fieldData_scales.txt',ds_scaler.scale_)
data_norm = ds_scaler.transform(df)   
np.savetxt('./fieldData_norm.txt', data_norm)

#X = df[["G1", "G2", "G3", "G4", "G5", "G6", "S1", "S2", "S3", "S4", "S5", "S6"]]
#y = df["nut"]

#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
