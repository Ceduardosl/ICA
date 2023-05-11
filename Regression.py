#%%
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
from collections import namedtuple

class reg:
    #Classe com os variados métodos de regressão e avaliação do ajuste.

    def __init__(self) -> None:
        pass

    def linreg (x, y, bias):
        results_linreg = namedtuple("linreg", ["slope", "intercept"])
        #Se bias == True, normalização é N
        #Se bias == False, normalização é N-1
        cov_matrix = np.cov(x, y, bias = bias)
        cov_x, cov_xy = cov_matrix[0, 0], cov_matrix[0, 1]
        slope = cov_xy/cov_x
        intercept = y.mean() - slope*x.mean()

        return (results_linreg(slope = slope, intercept = intercept))
    
    def evaluation_linreg (x, y, slope, intercept):
        evaluation_reg = namedtuple("evaluation", ["Pearson", "R2"])
        y_reg = slope*x + intercept
        pearson = np.corrcoef(y, y_reg)[0,1]
        R2 = np.power(pearson, 2)

        return (evaluation_reg(Pearson = pearson, R2 = R2))
    
    def mreg(X, y):
        result_mreg = namedtuple("mreg", ["Coef_Matrix", "y_mod"])
        if X.shape[0] != X.shape[1]:
            Coef_Matrix = np.linalg.lstsq(X,y)[0]
        else:
            Coef_Matrix = np.linalg.solve(X,y)[0]

        y_mod = np.dot(X, Coef_Matrix)

        return (result_mreg(Coef_Matrix = Coef_Matrix, y_mod = y_mod))

    def polyfit(x, y, p):
        result_polyfit = namedtuple("polyfit", ["Coef_Matrix", "y_mod"])
        X = np.ndarray(shape = (x.shape[0],p+1), dtype = "float64")
        for i in range(0, p+1):
            X[:,i] = np.power(x, i)
        if X.shape[0] != X.shape[1]:
            Coef_Matrix = np.linalg.lstsq(X,y)[0]
        else:
            Coef_Matrix = np.linalg.solve(X,y)[0]
        
        y_mod = np.dot(X, Coef_Matrix)

        #Coeft_matrix order -> [x^n, x^n-1, ... , x^0]
        return (result_polyfit(Coef_Matrix = Coef_Matrix, y_mod = y_mod))

#%%
if __name__ == '__main__':
    data = np.loadtxt("Dados/fish_perch.dat")
    x = data[:,1]
    y = data[:,0]
    N = len(data)

    linreg = reg.linreg(x,y,True)
    evt_linreg = reg.evaluation_linreg(x,y, linreg.slope, linreg.intercept)

    fig, ax = plt.subplots(dpi = 600)
    ax.scatter(x,y)
    ax.plot(x, linreg.slope*x + linreg.intercept, c = "red", label = "linreg", zorder = 2)
    ax.legend(loc = "lower right")
    ax.annotate("Pearson Correl = {:.3f}\nR² = {:.3f}".format(
        evt_linreg.Pearson,
        evt_linreg.R2), (0.015,0.88), xycoords = "axes fraction")

# %%

if __name__ == '__main__':
    data = np.loadtxt("Dados/fish_perch.dat")
    y = data[:,0]
    X = data[:, 1:]
    
    m_linreg = reg.mreg(X , y)
    
    fig, ax = plt.subplots(dpi = 600)
    ax.scatter(y, m_linreg.y_mod, zorder = 2)
    ax.plot(y, y, c = "black", zorder = 1)
    ax.set_ylabel("Dados Modelados")
    ax.set_xlabel("Dados Observados")

# %%
if __name__ == '__main__':
    data = np.loadtxt("Dados/fish_perch.dat")
    y = data[:,0]
    x = data[:,1]
    p = 6

    p_fit = reg.polyfit(x,y,p)
    fig, ax = plt.subplots(dpi = 600)
    ax.scatter(x,y, c = "red")
    ax.set_ylabel("Peso")
    ax.set_xlabel("Comprimento Vertical")
    ax.set_xlim(0, 45)
    ax.plot(x, p_fit.y_mod, zorder = 2, lw = 2)
    ax.set_title("Regressão Polinomial de {}° Ordem".format(p), loc = "left")
        
    
# %%
