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
#%%
# O código está utilizando a orientação nxp
# n é o número de amostras, p o número de característica
# As operações matriciais apresentam ordem contrária ao apresentando nas notas de aula
# Y = W*X (Nota de Aula) - Y = X*W (Presente Código)

X, Y = MNIST_data.get_train_data()

if np.linalg.matrix_rank(X) == min(X.shape):
    print("Matrix de dados de Posto Completo")
else:
    print("Matriz de dados de Posto Incompleto")

Y = one_hot_enconding(Y, 10)

if X.shape[0] != X.shape[1]:
    W = np.linalg.lstsq(X,Y)[0]
    # W = np.dot(np.linalg.pinv(X), Y)
else:
    W = np.linalg.solve(X,Y)[0]
#%%

X_test, Y_test = MNIST_data.get_test_data()
Y_test = one_hot_enconding(Y_test, 10)

Y_mod = np.dot(X_test, W)

count_ok = 0
for i in range(Y_mod.shape[0]):
    if Y_mod[i,:].argmax() == Y_test[i,:].argmax():
        count_ok += 1

print("Taxa de acerto na fase de teste = {:.2%}".format(count_ok/Y_mod.shape[0]))
print("Taxa de erro na fase de teste = {:.2%}".format(1-(count_ok/Y_mod.shape[0])))
#%%






a = np.reshape(x[10,:], (28, 28))
fig, ax = plt.subplots(dpi = 600)
ax.imshow(a, cmap = "Greys")
# %%
