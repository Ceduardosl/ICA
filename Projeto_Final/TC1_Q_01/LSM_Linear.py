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
#%%
x, y = MNIST_data.get_train_data()


#%%






a = np.reshape(x[10,:], (28, 28))
fig, ax = plt.subplots(dpi = 600)
ax.imshow(a, cmap = "Greys")
# %%
