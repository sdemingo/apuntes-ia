import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoide(x): # Derivada
    s = sigmoide(x)
    return s * (1 - s)

x = np.linspace(-30, 30, 100)

y1 = sigmoide(x)
y2 = d_sigmoide(x)  

plt.plot(x, y1, label="y = sigmoide")
plt.plot(x, y2, label="y = derivada")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoide y derivada")
plt.legend()
plt.grid(True)
plt.show()
