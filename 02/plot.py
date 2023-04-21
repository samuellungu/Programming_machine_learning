import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.axis([0, 35, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizza", fontsize=15)
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()
