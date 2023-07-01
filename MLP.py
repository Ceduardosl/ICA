#%%
import numpy as np
import matplotlib.pyplot as plt

def act_fun(u, fun):

    if fun == "step":
        u[np.where(u >= 0)] = 1
        u[np.where(u < 0)] = 0

    elif fun == "tanh":
        u = np.array(list(map(lambda x: (1-np.exp(-x))/(1+np.exp(-x)), u)))
    
    elif fun == "log":
        u = np.array(list(map(lambda x: 1/(1+np.exp(-x)), u)))
    
    return u

#%%
#data = p (features) x n (samples)
# fun_type = ["tanh", "step", "log"]
fun_type = "tanh"
# fun_type = "step"
# fun_type = "log"

i_data = np.loadtxt("Dados/wine_input.txt")
o_data = np.loadtxt("Dados/wine_target.txt")
i_data = (i_data - i_data.mean(axis = 0))/(i_data.std(axis = 0))

if (act_fun == "tanh"):
    #a codificação da saída fica -1 e 1 para tangente hiperbólica
    o_data[o_data == 0] = -1


eta = 0.01
Ne = 50
Nr = 10
tx_ok = np.empty((Nr))
best_dict = {"Acc_Best": 0, "W_best": 0}
#%%

