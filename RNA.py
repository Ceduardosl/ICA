#%%
import numpy as np
import matplotlib.pyplot as plt
def fun_sign(u):

    u[np.where(u >= 0)] = 1
    u[np.where(u < 0)] = 0
    
    return u
#%%
#data = pxn
i_data = np.loadtxt("Dados/derm_input.txt")
o_data = np.loadtxt("Dados/derm_target.txt")
eta = 0.01
Ne = 100
Nr = 10
tx_ok = np.empty((1, Nr))
#%%
for r in range(Nr):
    rand_index = np.random.permutation(i_data.shape[1])

    #Embaralhamento a cada rodada
    i_data = i_data[:, rand_index]
    o_data = o_data[:, rand_index]

    spt_point = 0.8 #split point 100*spt_point%
    t_input = i_data[:,0:int(spt_point*i_data.shape[1])]
    t_output = o_data[:,0:int(spt_point*i_data.shape[1])]
    v_input = i_data[:,int(spt_point*i_data.shape[1]):]
    v_output = o_data[:,int(spt_point*i_data.shape[1]):]

    #Definindo arquitetura
    #q = número de neuronios
    #Inicialização Aleatória
    #Inicialização Aleatória
    #W = q x (p+1)

    W = np.random.rand(v_output.shape[0], t_input.shape[0]+1)

    #Treino
    RMSE_plot = []
    for ep in range(Ne):
        rand_index = np.random.permutation(t_input.shape[1])
        t_input = t_input[:,rand_index]
        t_output = t_output[:,rand_index]
        RMSE = 0
        for i in range(t_input.shape[1]):
            X = np.append(-1, t_input[:,i])
            U = np.dot(W, X)
            Y = fun_sign(U)
            err = t_output[:,i] - Y

            X = np.expand_dims(X, 1)
            err = np.expand_dims(err, 1)
            RMSE = RMSE + 0.5*np.power(err, 2).sum()
            W = W + eta*np.dot(err, X.T)
        
        RMSE_plot.append(RMSE)
        print("Epoca = {}: RMSE = {:.2f}".format(ep, RMSE))

        #Validação
    count = 0
    for j in range(v_input.shape[1]):
        X = np.append(-1, v_input[:,j])
        U = np.dot(W, X)
        Y = fun_sign(U)
        # err = v_output[:,j] - Y
        if v_output[:,j].argmax() == Y.argmax():
            count += 1

    # tx_ok.append(count/v_input.shape[1])
    tx_ok[:,r] = count/v_input.shape[1]

# %%
fig, ax = plt.subplots(dpi = 600)
ax.plot(RMSE_plot)
ax.set_ylabel("RMSE")
ax.set_xlabel("Step")
# %%
