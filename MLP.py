#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def act_fun(u, fun):
    action_fun = namedtuple("act_fun", ["f", "df"])
    if fun == "step":
        u[np.where(u >= 0)] = 1
        u[np.where(u < 0)] = 0
        du = np.nan
    elif fun == "tanh":
        u = np.array(list(map(lambda x: (1-np.exp(-x))/(1+np.exp(-x)), u)))
        du = 0.5*(1-np.power(u,2))
    elif fun == "log":
        u = np.array(list(map(lambda x: 1/(1+np.exp(-x)), u)))
        du = u*(1-u)
    return (action_fun(f = u, df = du))
    
#%%
#data = p (features) x n (samples)
# fun_type = ["tanh", "step", "log"]
fun_type = "tanh"
# fun_type = "step"
# fun_type = "log"

i_data = np.loadtxt("Dados/derm_input.txt")
o_data = np.loadtxt("Dados/derm_target.txt")
i_data = (i_data - i_data.mean(axis = 0))/(i_data.std(axis = 0))

if (act_fun == "tanh"):
    #a codificação da saída fica -1 e 1 para tangente hiperbólica
    o_data[o_data == 0] = -1

eta = 0.01
Ne = 50
Nr = 1
tx_ok = np.empty((Nr))
spt_point = 0.8 #split point 100*spt_point%
best_dict = {"Acc_Best": 0, "W_best": 0}

#Arquitetura Rede
#2 Camadas Ocultas
#(p, q1, q2, c) =
#%%
RMSE_r = []
for r in range(Nr):
    rand_index = np.random.permutation(i_data.shape[1])
    i_data = i_data[:,rand_index]
    o_data = o_data[:,rand_index]

    t_input = i_data[:,0:int(spt_point*i_data.shape[1])]
    t_output = o_data[:,0:int(spt_point*o_data.shape[1])]
    v_input = i_data[:,int(spt_point*i_data.shape[1]):]
    v_output = o_data[:,int(spt_point*o_data.shape[1]):]

    dict_q = {
    0: 20,
    1: 10,
    2: v_output.shape[0]
    }
    dict_W = {}
    for j in range(len(dict_q)):
        if j == 0:
            dict_W[j] = np.random.rand(dict_q[j], t_input.shape[0] + 1)
        else:
            dict_W[j] = np.random.rand(dict_q[j], dict_q[j-1] + 1)

    RMSE_ep = []

    for ep in range(Ne):
        rand_index = np.random.permutation(t_input.shape[1])
        t_input = t_input[:,rand_index]
        t_output = t_output[:,rand_index]
        RMSE = 0
        for i in range(t_input.shape[1]):
            X = np.append(-1, t_input[:,i])
            for k in range(len(dict_q)):
                U = np.dot(dict_W[k], X)
                Y = act_fun(U, fun_type)
                if k == (len(dict_q)-1):
                    err = t_output[:,i] - Y.f
                    RMSE = RMSE + 0.5*np.power(err, 2).sum()
                    err = np.expand_dims(np.multiply(err, Y.df), 1)
                    X = np.expand_dims(X, 1)
                    dict_W[k] = dict_W[k] + np.dot(err, X.T)
                else:
                    X = np.append(-1, Y.f)
        RMSE_ep.append(RMSE/t_input.shape[1])
    RMSE_r.append(RMSE_ep)
fig, ax = plt.subplots(dpi = 600)
ax.plot(RMSE_ep)
#%%

rand_index = np.random.permutation(i_data.shape[1])
i_data = i_data[:,rand_index]
o_data = o_data[:,rand_index]

t_input = i_data[:,0:int(spt_point*i_data.shape[1])]
t_output = o_data[:,0:int(spt_point*o_data.shape[1])]
v_input = i_data[:,int(spt_point*i_data.shape[1]):]
v_output = o_data[:,int(spt_point*o_data.shape[1]):]

dict_q = {
    0: 20,
    1: 10,
    2: v_output.shape[0]
}
dict_W = {}
for i in range(len(dict_q)):
    if i == 0:
        dict_W[i] = np.random.rand(dict_q[i], t_input.shape[0]+1)
    else:
        dict_W[i] = np.random.rand(dict_q[i], dict_q[i-1] + 1)

#%%
X = np.append(-1, t_input[:,1])
for i in range(len(dict_q)):
    U = np.dot(dict_W[i], X)
    Y = act_fun(U, fun_type)
    if i == (len(dict_q)-1):
        print("OK")
        err = t_output[:,1] - Y.f
        RMSE = RMSE + 0.5*np.power(err, 2).sum()
        err = np.expand_dims(np.multiply(err, Y.df), 1)
        X = np.expand_dims(X, 1)
        print(X.shape)
        dict_W[i] = dict_W[i] + np.dot(err, X.T)
    else:
        X = np.append(-1, Y.f)
# %%
X = np.append(-1, t_input[:,1]) #add bias
U = act_fun(np.dot(dict_W[0], X), fun_type).f
U = np.append(-1, U)
Z = act_fun(np.dot(dict_W[1], U), fun_type).f
Z = np.append(-1, Z)
Y = act_fun(np.dot(dict_W[2], Z), fun_type).f
err1 = t_output[:,1] - Y
err1 = np.expand_dims(err1, 1)