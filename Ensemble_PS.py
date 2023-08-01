#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
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

def bagging_sampler(x, y):
    baggin_index = np.random.randint(0, x.shape[1], size = x.shape[1])

    x = x[:,baggin_index]
    y = y[:,baggin_index]

    return (x,y)

def initialize_weights(shape, Nm):
    W = {}
    for i in range(Nm):
        W[i] = np.random.rand(shape[0], shape[1])
    return W


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
Ne = 50
Nr = 5
Nm = 25
tx_ok = np.empty((Nr))
spt_point = 0.8
RMSE_ep = np.empty(0)

for r in range(Nr):

    rand_index = np.random.permutation(i_data.shape[1])

    #Embaralhamento a cada rodada
    i_data = i_data[:, rand_index]
    o_data = o_data[:, rand_index]

    X_train = i_data[:,0:int(spt_point*i_data.shape[1])]
    Y_train = o_data[:,0:int(spt_point*i_data.shape[1])]
    X_test = i_data[:,int(spt_point*i_data.shape[1]):]
    Y_test = o_data[:,int(spt_point*i_data.shape[1]):]


    count_m = np.full(Nm, np.nan)
    y_machines = np.empty(0)
    W = initialize_weights((Y_train.shape[0], X_train.shape[0]+1), Nm)

    for m in range(Nm):

        X, Y = bagging_sampler(X_train, Y_train)
        
        for e in range(Ne):

            rand_index = np.random.permutation(X.shape[1])
            X = X[:, rand_index]
            Y = Y[:, rand_index]
            RMSE = 0

            for i in range(X.shape[1]):
                x = np.append(-1, X[:,i])
                U = np.dot(W[m], x)
                y = act_fun(U, fun_type)
                err = Y[:, i] - y

                x = np.expand_dims(x, 1)
                err = np.expand_dims(err, 1)

                RMSE = RMSE + 0.5*np.power(err,2).sum()
                W[m] = W[m] + eta*np.dot(err, x.T)
        
        count_single = 0

        for j in range(X_test.shape[1]):
            x = np.append(-1, X_test[:,j])
            U = np.dot(W[m], x)
            y = act_fun(U, fun_type)

            if Y_test[:,j].argmax() == y.argmax():
                count_single += 1

        count_m[m] = count_single/X_test.shape[1]
        # print ("Máquina {}, rodada {}, Acerto {:.2%}".format(m+1, r, count/X_test.shape[1]))


    count_ensemble = 0
    for j in range(X_test.shape[1]):
        num = 0
        for m in range(Nm):
            x = np.append(-1, X_test[:,j])
            U = np.dot(W[m], x)
            y = act_fun(U, fun_type)
            num = num + y*count_m[m]

        y_m = num/count_m.sum()

        if Y_test[:,j].argmax() == y_m.argmax():
            count_ensemble += 1
        
    print ("Rodada {}, Acerto do Ensemble {:.2%}".format(r+1, count_ensemble/X_test.shape[1]))
#%%