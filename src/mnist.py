import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. PREPARACIÓN DE DATOS (Aplanamiento y Normalización)
# transforms.ToTensor() escala los píxeles de [0, 255] a [0, 1]
# transforms.Normalize ayuda a que ReLU no se sature al principio
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)



# 2. DEFINICIÓN DE LA ARQUITECTURA (El Blueprint que diseñamos)
class RedMNIST(nn.Module):
    def __init__(self):
        super(RedMNIST, self).__init__()
        self.flatten = nn.Flatten() # Convierte [28, 28] en [784] automáticamente
        self.stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)

model = RedMNIST()



# 3. CONFIGURACIÓN: Pérdida "Despiadada" y Optimizador Adam
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 4. BUCLE DE ENTRENAMIENTO (Vamos a hacer solo 2 épocas para que sea rápido)
print("Entrenando...")
for epoch in range(2):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()             # Reset de gradientes
        outputs = model(images)           # Forward
        loss = criterion(outputs, labels) # Error
        loss.backward()                   # Backpropagation
        optimizer.step()                  # Ajuste de pesos
        running_loss += loss.item()
    print(f"Época {epoch+1} - Error: {running_loss/len(trainloader):.4f}")



# 5. PRUEBA REAL: Vamos a ver qué opina la red de una imagen
dataiter = iter(trainloader)
images, labels = next(dataiter)

with torch.no_grad(): # Desactivamos gradientes para solo "adivinar"
    img = images[0]
    output = model(img)
    prediction = torch.exp(output).argmax() # Convertimos LogSoftmax a número real

plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.title(f"Predicción de la IA: {prediction.item()}")
plt.show()

