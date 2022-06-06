import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import datetime
def PCA(X , num_components):
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    return X_reduced
tstart = datetime.datetime.now()
data = pd.read_csv("eegData.csv",header=None)
x = data.iloc[:,0:16]
mat_reduced = PCA(x , 2)
tend = datetime.datetime.now()
print(mat_reduced.shape)
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2', s = 60 , palette= 'icefire')
plt.show()
delta=tend-tstart
print(delta)
