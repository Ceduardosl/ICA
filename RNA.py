#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def fun_output(u):
    if u < 0:
        output = 0
    if u >= 0:
        output = 1
    
    return output

d = np.array([0,1,1,1])
bias = 0
n = 0.5
w = np.array([bias, 0,0])
x = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]])
x = np.insert(x, 0, np.full(x.shape[1], -1), axis = 0)

#%%

print("Pesoas iniciais - w = {}\n".format(w))
count = 0
n_rod = 5
n_era = 2

for i in range(n_rod):
    print("Rodada {}".format(i))
    print("Vetor de Entrada {}".format(x[1:,:]))
    print("Saida Alvo {}\n".format(d))
    for j in range(n_era):
        print("Época {}".format(j+1))
        for k in range(x.shape[1]):
            count += 1
            u = np.dot(w, x[:,k].T)
            y = fun_output(u)

            print("iteração {} - w_old = {}".format(count, w))
            w = w + n*(d[k] - y)*x[:,k].T

            print("iteração {} - x = {}".format(count, x[:,k]))
            print("iteração {} - w_new = {}".format(count, w))
            print("iteração {} - u = {}".format(count, u))
            print("iteração {} - y = {}".format(count, y))
            print("iteração {} - erro = {}\n".format(count, d[k]-y))
    
    rand_index = np.random.permutation(x.shape[1])
    x = x[:, rand_index]
    d = d[rand_index]
# %%
g_x1, g_x2 = np.meshgrid(np.linspace(0,1, 100), np.linspace(0,1,100))

x1_arr, x2_arr = (g_x1.flatten(), g_x2.flatten())

X = np.vstack((x1_arr, x2_arr))
X = np.insert(X, 0, np.full(X.shape[1], -1), axis = 0)
y_list = []
for n in range(X.shape[1]):
    u = np.dot(w, X[:,n].T)
    y_list.append(fun_output(u))


X = np.insert(X, X.shape[0], y_list, axis = 0)

df = pd.DataFrame(X.T, columns = ["bias", "x1", "x2", "y"])
df.sort_values(by = ["x1", "x2"], ignore_index = True, inplace = True)
trans_list = []
for i in range(len(df)-1):
    if df.y.iloc[i+1] - df.y.iloc[i] == 1:
        trans_list.append([df.x1.iloc[i], df.x2.iloc[i]])
df_trans = pd.DataFrame(trans_list, columns = ["x1","x2"])
#%%
fig, ax = plt.subplots(dpi = 600)
ax.scatter(df.x1.loc[df.y == 0], df.x2.loc[df.y == 0], c = "green", label = "Class 1")
ax.scatter(df.x1.loc[df.y == 1], df.x2.loc[df.y == 1], c = "blue", label = "Class 2")
ax.plot(df_trans.x1, df_trans.x2, c = "red", label = "Decision Surface", lw = 3)
ax.scatter(0,0, c = "green", edgecolors = "black", s = 100)
ax.scatter([0,1,1], [1,0,1], c = "blue", edgecolors = "black", s = 100, zorder = 3)
ax.set_title("PS - Porta Lógica OR")
fig.savefig("teste.png", dpi = 600, bbox_inches = "tight", facecolor = "w")
# %%

# %%
