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


if (act_fun == "tanh"):
    #a codificação da saída fica -1 e 1 para tangente hiperbólica
    o_data[o_data == 0] = -1
#%%

eta = 0.01
# Ne = 50
# Nr = 5

Ne = 1
Nr = 10

tx_ok = np.empty((Nr))
spt_point = 0.8
RMSE_ep = np.empty(0)

#%%
q1 = 86

W = np.random.normal(loc = 0, scale = 0.1, size = (q1, i_data.shape[0]+1))
# W2 = np.random.normal(loc = 0, scale = 0.1, size = (o_data.shape[0], q1+1))
#%%
for r in range(Nr):
    rand_index = np.random.permutation(i_data.shape[1])

    i_data = i_data[:,rand_index]
    o_data = o_data[:,rand_index]

    X_train = i_data[:,0:int(spt_point*i_data.shape[1])]
    Y_train = o_data[:,0:int(spt_point*i_data.shape[1])]
    X_test = i_data[:,int(spt_point*i_data.shape[1]):]
    Y_test = o_data[:,int(spt_point*i_data.shape[1]):]

    Z = np.empty(0)

    for i in range(X_train.shape[1]):
        x = np.append(-1, X_train[:,i])
        U = np.dot(W, x)
        z = act_fun(U, fun_type)
        z = np.append(-1, z)
        Z = np.append(Z, z)
    
    Z = Z.reshape((X_train.shape[1], q1+1))

    M = np.dot(Y_train, np.linalg.pinv(Z.T))
    
    count = 0
    for j in range(X_test.shape[1]):
        x = np.append(-1, X_test[:,j])
        U1 = np.dot(W, x)
        z = act_fun(U1, fun_type)
        z = np.append(-1, z)
        y = np.dot(M, z)

        if Y_test[:,j].argmax() == y.argmax():
            count += 1

    print("Rodada {}, Taxa de acerto {:.2%}".format(r+1, count/X_test.shape[1]))
#%%