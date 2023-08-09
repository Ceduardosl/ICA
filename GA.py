import numpy as np
import matplotlib.pyplot as plt
import time

def input_matrix(x, order):
    X = np.empty((x.shape[0], order + 1))
    for i in range(order+1):
        if i == 0:
            X[:,i] = 1
        else:
            X[:,i] = np.power(x,i)
    
    return X

def Hardamad_Prod(a, b):
    
    prod = np.array([np.multiply(x,y) for x, y in zip(a,b)])

    return prod

def EQM(M, y_true, X):

    coef = np.power(y_true - np.dot(X, M), 2).sum()

    return coef #Minimização, por isso 1/coef

def Elitismo(M, n_el):

    return M[0:n_el]

def Selecao(M, FO, n_el, nsel_torn):
    
    M_sel = []
    for i in range(nsel_torn):
        j = np.random.randint(n_el, M.shape[0]) 
        k = np.random.randint(n_el, M.shape[0])
        #(n_el, M.shape[0]) Selecionar dentre os q não foram para o elitismo
        if FO[j] >= FO[k]:
            M_sel.append(M[j])
        else:
            M_sel.append(M[k])

    M_sel = np.array(M_sel)
    
    return M_sel

def Cruzamento(M, pc, n_off, y, X):
    
    M_off = []
    for i in range(0, int(n_off/2)):

        j = np.random.randint(0, M.shape[0])
        k = np.random.randint(0, M.shape[0])

        if np.random.uniform(0,1) <= pc:
            off_aux = np.array([
                M[j] + M[k],
                1.5*M[j] - 0.5*M[k],
                -0.5*M[j] + 1.5*M[k]
            ])

            FO_off = np.apply_along_axis(EQM, 1, off_aux, y, X)
            sorted_index = np.argsort(FO_off)
            off_aux = off_aux[sorted_index]
            M_off.append(off_aux[0])
            M_off.append(off_aux[1])

        else:
            M_off.append(M[j])
            M_off.append(M[k])
    
    M_off = np.array(M_off)

    return M_off

def Mutacao(M_off, pm, m_inc, upper_limit, lower_limit):
    M_mut = np.empty(shape = M_off.shape)

    for i in range(M_off.shape[0]):
        if np.random.uniform(0,1) <= pm:
            mut = m_inc*(upper_limit - lower_limit)*np.random.normal(0,1,size = (1,M_off.shape[1]))
            M_mut[i,:] = M_off[i,:] + mut
        else:
            M_mut[i,:] = M_off[i,:]
    
    M_mut[M_mut > upper_limit] = upper_limit
    M_mut[M_mut < lower_limit] = lower_limit

    return M_mut

data = np.genfromtxt("aerogerador.dat")
x = data[:,0]
y = data[:,1]

Ns = 5000
Nr = 3

k = 4
p = 22 #Num elementos
l_limit, u_limit = -6, 2

pc, pm = 0.95, 0.05 #prob de cruzamento e mutação
m_inc = 0.2 #Incremento da mutação
n_el = 2 #Número de indivíduos selecionados no elitismo
nsel_torn = 10 #Número de selecionados no torneio
n_off = 10 #Número de novos indivíduos gerados no cruzamento

X = input_matrix(x, order = k)

fig, ax = plt.subplots()
fig_b, ax_b = plt.subplots()
tic = time.perf_counter()

for r in range(Nr):
    EQM_best = []
    for s in range(Ns):
        
        if s == 0:
            Mi_cand = np.random.uniform(l_limit, u_limit, size = (p, k+1))
            FO_cand = np.apply_along_axis(EQM, 1, Mi_cand, y, X)

            sorted_index = np.argsort(FO_cand)
            Mi_cand, FO_cand = Mi_cand[sorted_index], FO_cand[sorted_index]
            EQM_best.append(FO_cand[0])

        else:
            M_el = Elitismo(Mi_cand, n_el)
            M_sel = Selecao(Mi_cand, FO_cand, n_el, nsel_torn)
            M_off = Cruzamento(Mi_cand, pc, n_off, y, X)
            M_off = Mutacao(M_off, pm, m_inc, u_limit, l_limit)

            Mi_cand = np.concatenate((M_el, M_sel, M_off))
            FO_cand = np.apply_along_axis(EQM, 1, Mi_cand, y, X)
            sorted_index = np.argsort(FO_cand)
            Mi_cand, FO_cand = Mi_cand[sorted_index], FO_cand[sorted_index]
            EQM_best.append(FO_cand[0])

    ax.plot(EQM_best)
    ax_b.plot(x, np.dot(X, Mi_cand[0]), label = "PSO - Rodada {}".format(r+1), lw = 2, zorder = 2)
        # print ("iteração {} - EQM do melhor indivíduo = {:.1f}".format(s+1, FO_cand[0]))
toc = time.perf_counter()
print("\nTempo de Simulação = {:.3f} segundos, Número de iterações = {}, Número de Rodadas = {}".format(toc-tic, Ns, Nr))

ax.set_yscale("log")
ax.set_ylabel("EQM")
ax.set_xlabel("Iteração");

ax_b.scatter(x,y, c = "brown", label = "Dados Observados", zorder = 1)
ax_b.set_xlabel("Velocidade do Vento (m/s)")
ax_b.set_ylabel("Potência Gerada (kW)")
ax_b.set_title("Melhor indivíduo em cada uma das {} rodadas".format(Nr), loc = "left")
ax_b.legend();
