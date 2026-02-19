import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()

        self.dropout = nn.Dropout(0.5)  # Config del dropout

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
        x = self.dropout(x) # Las neuronas de fc1 se "arriesgan" a ser apagadas
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)



###
# Preparación de los datos y el dataset
##

# 1. Definimos la transformación: 
# Convertir a Tensor y normalizar (Media 0.5, Desviación 0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Descargamos el Dataset de entrenamiento y el de prueba
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

# 3. Creamos los Loaders: Los "grifos" que entregan 64 imágenes a la vez
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)





####
## Entrenamiento
###


model = NetCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


epochs = 5

for epoch in range(epochs):
    model.train() # Modo entrenamiento (activa el Dropout)
    running_loss = 0.0
    
    for images, labels in trainloader:
        # Reset de gradientes
        optimizer.zero_grad()
        
        # Forward: Ya no aplanamos 'images', la CNN quiere el 28x28
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward y Optimización
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Época {epoch+1} - Error medio: {running_loss/len(trainloader):.4f}")



# 3. Evaluación final
model.eval() # Modo evaluación (DESACTIVA el Dropout para usar toda la potencia de la red)
# Aquí añadirías el código de testeo con el testloader
