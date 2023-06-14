#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from collections import namedtuple

def PCA(arr):
    #Orientação do vetor deve ser:
    #variáveis (features) nas linhas
    #indivíduos (samples) nas colunas
    #Orientação contrária ao exigido pelo Scikit Learn
    results_PCA = namedtuple("PCA", ["components",
                                    "variance",
                                    "explained_variance_ratio",
                                    "x_transf"])
    x_arr = arr - np.repeat(arr.mean(axis = 1).reshape(-1,1),
                        repeats = i_data.shape[1], axis = 1)
    cov_x = np.cov(x_arr)
    print(cov_x.shape)
    w, v = np.linalg.eig(cov_x)
    sorted_idx = w.argsort()[::-1]
    w = w[sorted_idx]
    v = v[:, sorted_idx]

    z = np.dot(v.T, x_arr)

    ve = np.cumsum(w)/w.sum()

    return (results_PCA(components = v,
                        variance = w,
                        explained_variance_ratio = ve,
                        x_transf = z))
#%%
i_data = np.loadtxt("Dados/derm_input.txt")

myPCA = PCA(i_data)
# %%
sk_PCA = PCA(n_components = 5)
sk_PCA.fit(i_data.T)
#%%
fig, ax = plt.subplots(dpi = 600)
ax.plot(myPCA.explained_variance_ratio, label = "Cumulative Explained Variance", c = "red")
ax.bar(x = range(0, len(myPCA.explained_variance)),
    height = myPCA.explained_variance/myPCA.explained_variance.sum(),
    label = "Explained Variance")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("PCA index")
ax.set_ylabel("Explained variance ratio")
ax.legend(loc = "center right")
# %%
