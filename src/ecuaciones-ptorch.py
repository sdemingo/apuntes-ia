import torch
import numpy as np


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



# Uso:
X, y = generar_datos(500)
print (X)
