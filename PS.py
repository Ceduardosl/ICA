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

i_data = np.loadtxt("Dados/derm_input.txt")
o_data = np.loadtxt("Dados/derm_target.txt")
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
            X = np.append(-1, t_input[:,i]) #add bias
            U = np.dot(W, X)
            Y = act_fun(U, fun_type)
            err = t_output[:,i] - Y

            X = np.expand_dims(X, 1)
            err = np.expand_dims(err, 1)
            RMSE = RMSE + 0.5*np.power(err, 2).sum()
            W = W + eta*np.dot(err, X.T)
        
        RMSE_plot.append(RMSE)

        #Validação
    count = 0
    for j in range(v_input.shape[1]):
        X = np.append(-1, v_input[:,j])
        U = np.dot(W, X)
        Y = act_fun(U, fun_type)
        # err = v_output[:,j] - Y
        if v_output[:,j].argmax() == Y.argmax():
            count += 1
    
    tx_ok[r] = count/v_input.shape[1]
    
    if tx_ok[r] >= best_dict["Acc_Best"]:
        best_dict["Acc_Best"] = tx_ok[r]
        best_dict["W_best"] = W
# %%
fig, ax = plt.subplots(dpi = 600)
ax.plot(RMSE_plot)
ax.set_ylabel("RMSE")
ax.set_xlabel("Step")
# %%
if fun_type == "step":
    step_RMSE = RMSE_plot
elif fun_type == "tanh":
    tanh_RMSE = RMSE_plot
elif fun_type == "log":
    log_RMSE = RMSE_plot
#%%

fig, ax = plt.subplots(dpi = 600)
ax.plot(np.arange(-10, 10, 0.1), act_fun(np.arange(-10, 10, 0.1), "step"), c = "red", label = "Step")
ax.plot(np.arange(-10, 10, 0.1), act_fun(np.arange(-10, 10, 0.1), "tanh"), c = "blue", label  ="tanH")
ax.plot(np.arange(-10, 10, 0.1), act_fun(np.arange(-10, 10, 0.1), "log"), c = "green", label = "Sigmoide")
ax.legend()
# %%
fig, ax = plt.subplots(dpi = 600)
ax.plot(step_RMSE, c = "red", label = "Step")
ax.plot(tanh_RMSE, c = "blue", label = "tanH")
ax.plot(log_RMSE, c = "green", label = "Sigmoide")
ax.legend()
# %%
