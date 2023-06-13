#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#%%
i_data = np.loadtxt("Dados/derm_input.txt")
# i_data = np.loadtxt("Dados/wine_input.txt")
#%%
x_arr = i_data - np.repeat(i_data.mean(axis = 1).reshape(-1,1),
                        repeats = i_data.shape[1], axis = 1)

cov_x = np.cov(x_arr)

#w autovalor e v autovetor
#sorted_index = w.argsort()[::-1] 
#index da ordem crescente dos autovalores, utilizo para ordenar os vetores w e v.

w, v = np.linalg.eig(cov_x)
sorted_idx = w.argsort()[::-1] #ordem crescente dos autovalores,
w = w[sorted_idx]
v_s = v[:,sorted_idx]

Q = v.T
z = np.dot(Q,x_arr)
cov_z = np.cov(z)

vt = w.sum()
ve = np.cumsum(w)/vt

# %%
fig, ax = plt.subplots(dpi = 600)
ax.plot(ve, label = "Cumulative Explained Variance", c = "red")
ax.bar(x = range(0, len(w)), height = w/vt, label = "Explained Variance")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("PCA index")
ax.set_ylabel("Explained variance ratio")
ax.legend(loc = "center right")
# %%
