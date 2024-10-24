import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


log = pd.read_csv('./training_log_final.csv')
epoch = np.arange(1,log.shape[0]+1)

fontSize = 21
plt.figure(figsize=(10,7))
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
plt.plot(epoch, log['Train Loss'], 'r--')
#plt.plot(epoch, log['Val Loss'], 'r-')
plt.plot(epoch, log['Train R^2'], 'r:')
#plt.plot(epoch, log['Val R^2'], 'b-')
plt.xlabel('epoch', fontsize=fontSize)
plt.xticks(fontsize=fontSize)  
plt.yticks(fontsize=fontSize)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(fontSize)  
ax.yaxis.get_offset_text().set_fontsize(fontSize)
plt.legend(['Loss', 'R2 Score'], fontsize=fontSize, frameon=False)
plt.savefig('./Results/log.png')