#%%
import numpy as np
from mnist import MNIST #https://pypi.org/project/python-mnist/#description
import matplotlib.pyplot as plt
import time

class MNIST_data():

    def __init__(self):
        pass
    
    def get_train_data():
        
        mndata = MNIST("./")
        mndata.gz = True
        x, y = mndata.load_training()

        return (np.array(x), np.array(y))
    
    def get_test_data():
        
        mndata = MNIST("./")
        mndata.gz = True
        x, y = mndata.load_testing()

        return (np.array(x), np.array(y))

def one_hot_enconding(y, n):

    y_enc = np.zeros((y.shape[0], n))
    for i in range(y.shape[0]):
        y_enc[i, y[i]] = 1

    return y_enc

def norm_data(df):
    df_std = np.zeros(df.shape)

    for i in range(df.shape[0]):
        # df_std[i,:] = (df[i,:] - df[i,:].mean())/(df[i,:].std(ddof = std_ddof))
        df_std[i,:] = (df[i,:] - df[i,:].min())/(df[i,:].max() - df[i,:].min())
    return df_std

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
# O código está utilizando a orientação nxp
# n é o número de amostras, p o número de característica
# As operações matriciais apresentam ordem contrária ao apresentando nas notas de aula
# Y = W*X (Nota de Aula) - Y = X*W (Presente Código)

X, Y = MNIST_data.get_train_data()
X_test, Y_test = MNIST_data.get_test_data()
    
if np.linalg.matrix_rank(X) == min(X.shape):
    print("Matrix de dados de Posto Completo")
else:
    print("Matriz de dados de Posto Incompleto")

Y = one_hot_enconding(Y, 10)
Y_test = one_hot_enconding(Y_test, 10)

X = norm_data(X)
X_test = norm_data(X_test)
#%%
Nr = 1
fun_type = "log"
eta = 0.01
Ne = 50
Nr = 1
tx_ok = np.empty((Nr))
best_run = {"Acc_Best": 0, "W_best": 0}

if fun_type == "tanh":
    #codificação da saída fica -1 e 1 para tangente hiperbólica
    Y[Y == 0] = -1
    Y_test[Y == 0] = -1
#%%
t = time.process_time()
for r in range(Nr):
    #Não embaralhei a cada rodada, pois os dados de treino sempre serão X e Y
    #Assim, como não haverá split do conjunto em dados de treino e teste
    #Decidi embaralhar só dentro de cada época
    #Inicialização aleatória dos pesos

    W = np.random.rand(Y.shape[1], X.shape[1]+1)
    RMSE_ep = np.full(Ne, np.nan)
    
    for ep in range(Ne):
        #Embaralhamento da matriz de dados saída
        rand_index = np.random.permutation(X.shape[0])
        X = X[rand_index, :]
        Y = Y[rand_index, :]
        RMSE = 0
        for i in range(X.shape[0]):
            x = np.append(-1, X[i,:]) #add bias
            U = np.dot(W, x)
            y = act_fun(U, fun_type)
            err = Y[i,:] - y

            x = np.expand_dims(x, 1)
            err = np.expand_dims(err, 1)

            RMSE = RMSE + 0.5*np.power(err, 2).sum()
            W = W + eta*np.dot(err, x.T)
        
        RMSE_ep[ep] = RMSE/X.shape[0]

elapsed_time = time.process_time() - t


#%%
count = 0
for j in range(X_test.shape[0]):
    x = np.append(-1, X_test[j,:])
    U = np.dot(W, x)
    y = act_fun(U, fun_type)

    if Y_test[j,:].argmax() == y.argmax():
        count += 1
print("{:.2%}".format(count/X_test.shape[0]))
# %%
