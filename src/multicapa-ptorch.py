import torch
import torch.nn as nn
import torch.optim as optim

# Datos de entrada (XOR) y resultados esperados (Y)
# Ahora para los datos usamos tensores y no matrices de numpy como
# antes
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class RedXOR(nn.Module):
    def __init__(self):
        super(RedXOR, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2,4),  # Entrada: 2 neuronas (los bits), Oculta: 4 neuronas
            nn.Sigmoid(),
            nn.Linear(4, 1), # Entrada: 4 neuronas de la oculta; Salida: 1 neurona (el resultado 0 o 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)


model = RedXOR()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

epochs=5000

print("Entrenando ...")

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion (outputs, y)      # Error

    optimizer.zero_grad()
    loss.backward()                    # Backpropagation
    optimizer.step() 

    if (epoch + 1) % 500 == 0:
        print(f"Época {epoch+1}/{epochs} - Error: {loss.item():.4f}")




