import numpy as np
from neuralNetworks import neuralNetwork
from mat4py import loadmat
data = loadmat('ex4data1.mat')
W1 = np.random.randn(25,401)
print(W1)
