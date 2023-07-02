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
Ne = 2
Nr = 10
tx_ok = np.empty((Nr))
spt_point = 0.8 #split point 100*spt_point%
best_dict = {"Acc_Best": 0, "W_best": 0}

#Arquitetura Rede
#2 Camadas Ocultas
#(p, q1, q2, c) =

p = 5
q1 = 13
q2 = 13
#%%

rand_index = np.random.permutation(i_data.shape[1])
i_data = i_data[:,rand_index]
o_data = o_data[:,rand_index]

t_input = i_data[:,0:int(spt_point*i_data.shape[1])]
t_output = o_data[:,0:int(spt_point*o_data.shape[1])]
v_input = i_data[:,int(spt_point*i_data.shape[1]):]
v_output = o_data[:,int(spt_point*o_data.shape[1]):]
#%%

W_i = np.random.rand(v_output.shape[0], t_input.shape[0]+1) 

#%%
dict_W = {}
for i in range(2):
    dict_W[i+1] = W_i

# %%
RMSE_plot = []
for ep in range(Ne):
    rand_index = np.random.permutation(t_input.shape[1])
    t_input = t_input[:,rand_index]
    t_output = t_output[:,rand_index]
    RMSE = 0
    for i in range(t_input.shape[1]):
        X = np.append(-1, t_input[:,i]) #add bias
        U = np.dot(W, X)
        Y = act_fun(U, fun_type)
        err = t_output[:,i] - Y

        X = np.expand_dims(X, 1)
        err = np.expand_dims(err, 1)
        RMSE = RMSE + 0.5*np.power(err, 2).sum()
        W = W + eta*np.dot(err, X.T)