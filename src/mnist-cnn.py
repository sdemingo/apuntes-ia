import torch
import torch.nn as nn
import torch.nn.functional as F

class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        
        # 1. CAPAS CONVOLUCIONALES (Extracción de rasgos)
        # Entrada: 1 canal (BN). Salida: 32 canales (filtros).
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 

        # Entrada: 32 canales. Salida: 64 canales.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Capa de reducción (Pooling)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2. CAPAS LINEALES (Clasificación)
        # El "5*5*64" sale de los cálculos de dimensiones tras las convoluciones
        self.fc1 = nn.Linear(5 * 5 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Aplicamos: Convolución -> ReLU -> Pooling
        # De 28x28 a 26x26 (Conv) -> a 13x13 (Pool)
        x = self.pool(F.relu(self.conv1(x)))
        
        # De 13x13 a 11x11 (Conv) -> a 5x5 (Pool)
        x = self.pool(F.relu(self.conv2(x)))
        
        # APLANADO (Flatten)
        # Pasamos de un bloque 3D (64 filtros de 5x5) a un vector plano
        x = x.view(-1, 5 * 5 * 64) 
        
        # Capas clásicas (XOR style)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Salida de 10 neuronas
        
        return F.log_softmax(x, dim=1)

model_cnn = NetCNN()
