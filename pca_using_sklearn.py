import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("eegData.csv",header=None)
data.keys()
scaling=StandardScaler()
scaling.fit(data)
Scaled_data=scaling.transform(data)
tstart = datetime.datetime.now()
principal=PCA(n_components=2)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
tend = datetime.datetime.now()
print(x.shape)
principal_df = pd.DataFrame(x, columns = ['PC1','PC2'])
plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2', s = 60 , palette= 'icefire')
plt.show()
delta=tend-tstart
print(delta)
