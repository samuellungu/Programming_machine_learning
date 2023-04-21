import numpy as np

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
X[0:5]