import numpy as np
import matplotlib as plt

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y)
plt.show()