import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

x = np.linspace(-10, 10, 100)
y1 = relu(x)
y2 = d_relu(x)  

plt.plot(x, y1, label="y = relu")
plt.plot(x, y2, label="y = derivada")

plt.xlabel("x")
plt.ylabel("y")
plt.title("ReLU y derivada")
plt.legend()
plt.grid(True)
plt.show()
