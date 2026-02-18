import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

def generar_datos(n_muestras=500):
    # 1. Generamos x1 y x2 al azar entre -10 y 10
    x_reales = np.random.uniform(-10, 10, (n_muestras, 2))

    inputs = []
    targets = []

    for i in range(n_muestras):
        x1, x2 = x_reales[i]

        # 2. Calculamos coeficientes (suponiendo a=1)
        a = 1.0
        b = -(x1 + x2)
        c = x1 * x2

        inputs.append([a, b, c])
        # Es importante ordenar x1 y x2 (ej. de menor a mayor)
        # para que la red no se confunda con el orden de las salidas
        targets.append(sorted([x1, x2]))

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)



# Generamos un data set de 500 ecuaciones donde X es un tensor con a,b
# y c e Y es un tensor con los resultados de esas ecuaciones
X, y = generar_datos(500)
print (X)


class RedEcuaciones(nn.Module):
    def __init__(self):
        super(RedEcuaciones, self).__init__()

        self.hidden1 = nn.Linear(3, 16)     # Capa de entrada (3) a primera capa oculta (16)
        self.hidden2 = nn.Linear(16, 16)    # Segunda capa oculta (16 a 16)
        self.output = nn.Linear(16, 2)      # Capa de salida (16 a 2 soluciones)
        self.relu = nn.ReLU()               # Función de activación

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x



model = RedEcuaciones()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs=8000

print("Entrenando ...")

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion (outputs, y)      # Error

    optimizer.zero_grad()
    loss.backward()         # Backpropagation
    optimizer.step()        # Ajuste pesos automático (de eso se encarga el optimizador)

    if (epoch + 1) % 500 == 0:
        print(f"Época {epoch+1}/{epochs} - Error: {loss.item():.4f}")
