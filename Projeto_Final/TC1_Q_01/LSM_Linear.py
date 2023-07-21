#%%
import numpy as np
from mnist import MNIST #https://pypi.org/project/python-mnist/#description
import matplotlib.pyplot as plt

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
#%%
# O código está utilizando a orientação nxp
# n é o número de amostras, p o número de característica
# As operações matriciais apresentam ordem contrária ao apresentando nas notas de aula
# Y = W*X (Nota de Aula) - Y = X*W (Presente Código)

Nr = 10
X, Y = MNIST_data.get_train_data()
X_test, Y_test = MNIST_data.get_test_data()
    
if np.linalg.matrix_rank(X) == min(X.shape):
    print("Matrix de dados de Posto Completo")
else:
    print("Matriz de dados de Posto Incompleto")

Y = one_hot_enconding(Y, 10)
Y_test = one_hot_enconding(Y_test, 10)

X = norm_data(X, 1)
X_test = norm_data(X_test, 1)
#%%
tx_ok = np.zeros(10)
for i in range(Nr):
    rand_index = np.random.permutation(X.shape[0])
    X = X[rand_index,:]
    Y = Y[rand_index,:]

    if X.shape[0] != X.shape[1]:
        W = np.linalg.lstsq(X,Y)[0]
        # W = np.dot(np.linalg.pinv(X), Y)
    else:
        W = np.linalg.solve(X,Y)[0]

    Y_mod = np.dot(X_test, W)

    count_ok = 0

    for j in range(Y_mod.shape[0]):
        if Y_mod[j,:].argmax() == Y_test[j,:].argmax():
            count_ok += 1
    
    tx_ok[i] = count_ok/Y_mod.shape[0]

print('''
Taxa de acerto Média = {:.2%} \n
Taxa de erro Média = {:.2%} \n
Melhor Taxa de Acerto = {:.2%} \n
Pior Taxa de Acerto = {:.2%} \n
Desv. Pad. Taxa de Acerto = {:.2%}
'''.format(tx_ok.mean(), 1-tx_ok.mean(),
    tx_ok.max(), tx_ok.min(), tx_ok.std()))

#%%


a = np.reshape(X[5,:], (28, 28))
fig, ax = plt.subplots(dpi = 600)
ax.imshow(a, cmap = "Greys")
# %%
