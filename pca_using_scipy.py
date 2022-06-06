import pandas as pd
def PCA(data, dims_rescaled_data=2):
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    data -= data.mean(axis=0)
    R = NP.cov(data, rowvar=False)
    evals, evecs = LA.eigh(R)
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :dims_rescaled_data]
    return NP.dot(evecs.T, data.T).T, evals, evecs

def plot_pca(data):
    from matplotlib import pyplot as MPL
    clr1 =  '#2026B2'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig, a= PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    MPL.show()

data = pd.read_csv("eegData.csv",header=None)
plot_pca(data)
