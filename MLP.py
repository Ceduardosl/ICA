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
        du = 0.5*(1-np.power(u,2)) + 0.05
    elif fun == "log":
        u = np.array(list(map(lambda x: 1/(1+np.exp(-x)), u)))
        du = u*(1-u) + 0.05
    return (action_fun(f = u, df = du))

def Hardamad_Prod(a, b):
    
    prod = np.array([np.multiply(x,y) for x, y in zip(a,b)])

    return prod

def std_data(df, std_ddof):
    
    for i in range(df.shape[0]):
        df[i,:] = (df[i,:] - df[i,:].mean())/(df[i,:].std(ddof = std_ddof))

    return df
#%%
#data = p (features) x n (samples)
# fun_type = ["tanh", "step", "log"]
# fun_type = "tanh"
# fun_type = "step"
fun_type = "log"

i_data = np.loadtxt("Dados/derm_input.txt")
o_data = np.loadtxt("Dados/derm_target.txt")

i_data = std_data(i_data, std_ddof = 1)

if (fun_type == "tanh"):
    #a codificação da saída fica -1 e 1 para tangente hiperbólica
    o_data[o_data == 0] = -1

eta = 0.05
Ne = 100
Nr = 10
mon = 0.75
tx_ok = np.empty((Nr))
spt_point = 0.8 #split point 100*spt_point%
best_dict = {"Acc_Best": 0, "W_0": 0, "W_1": 0, "W_2": 0}


#Arquitetura Rede
#1 camada oculta
#(p, q, c) =

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
    0: 15,
    1: v_output.shape[0]
    }

    W = {}
    for j in range(len(dict_q)):
        if j == 0:
            W[j] = np.random.rand(dict_q[j], t_input.shape[0] + 1)*0.01
        else:
            W[j] = np.random.rand(dict_q[j], dict_q[j-1] + 1)*0.01
    W_old = W.copy()
    RMSE_ep = []

    #Treino
    for ep in range(Ne):
        rand_index = np.random.permutation(t_input.shape[1])
        t_input = t_input[:,rand_index]
        t_output = t_output[:,rand_index]
        RMSE = 0
        for i in range(t_input.shape[1]):

            X = np.append(-1, t_input[:,i])
            U1 = np.dot(W[0], X)
            Z, dZ = act_fun(U1, fun_type)
            Z = np.append(-1, Z)
            U2 = np.dot(W[1], Z)
            Y, dY = act_fun(U2, fun_type)               

            err = t_output[:,i] - Y
            RMSE = RMSE + 0.5*np.power(err, 2).sum()

            err = np.expand_dims(Hardamad_Prod(err, dY), 1)
            X = np.expand_dims(X, 1)
            Z = np.expand_dims(Z, 1)
            
            DDi = Hardamad_Prod(dZ, np.dot(W[1][:,1:].T, err))

            W[0] = W[0] + eta*np.dot(Hardamad_Prod(dZ, np.dot(
                W[1][:,1:].T, err)), X.T) + mon*(W[0] - W_old[0])
            
            W[1] = W[1] + eta*np.dot(err, Z.T) + mon*(W[1] - W_old[1])

            W_old = W.copy()
        RMSE_ep.append(RMSE/t_input.shape[1])
    RMSE_r.append(RMSE_ep)

    #validação
    count = 0
    for v in range(v_input.shape[1]):
        X = np.append(-1, v_input[:,v])
        U1 = np.dot(W[0], X)
        Z = act_fun(U1, fun_type).f
        Z = np.append(-1, Z)
        U2 = np.dot(W[1], Z)
        Y = act_fun(U2, fun_type).f

        if v_output[:,v].argmax() == Y.argmax():
            count += 1

    tx_ok[r] = count/v_input.shape[1]

    if tx_ok[r] >= best_dict["Acc_Best"]:
        best_dict["Acc_Best"] = tx_ok[r]
        best_dict["W_0"] = W[0]
        best_dict["W_1"] = W[1]

#%%
Acc_stats = '''
    Acurácia:\n
     Média: {:.2%}\n
     Mediana: {:.2%}\n
     Desv. Pad.: {:.2%}\n
     Mínimo: {:.2%}\n
     Máximo: {:.2%}\n
    '''.format(tx_ok.mean(), np.median(tx_ok), tx_ok.std(),
        tx_ok.min(), tx_ok.max())
fig, ax = plt.subplots(dpi = 600)
for p in RMSE_r:
    ax.plot(p)
ax.set_title("MLP 1 camada oculta", loc = "left")
ax.set_ylabel("RMSE")
ax.set_xlabel("Epoch")
ax.annotate(Acc_stats, xycoords = "axes fraction", xy = (0.65, 0.30))
ax.axhline(y = 0, c = "black", ls = "--", lw = 0.75)
# %%