import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Nuestro pequeño dataset de ciudades
ciudades = ["MADRID", "BARCELONA", "VALENCIA", "SEVILLA", "ZARAGOZA",
            "MALAGA", "MURCIA", "PALMA", "BILBAO", "ALICANTE",
            "CORDOBA", "VALLADOLID", "VIGO", "GIJON", "GRANADA"]

alfabeto = sorted(list(set("".join(ciudades) + ".")))
char_to_int = {char: i for i, char in enumerate(alfabeto)}
int_to_char = {i: char for i, char in enumerate(alfabeto)}
n_letras = len(alfabeto)

def palabra_a_tensor(palabra):
    tensor_x = torch.zeros(len(palabra), 1, n_letras)
    for i, char in enumerate(palabra):
        tensor_x[i][0][char_to_int[char]] = 1
    return tensor_x

def palabra_a_objetivo(palabra):
    indices = [char_to_int[char] for char in palabra[1:]]
    indices.append(char_to_int['.'])
    return torch.LongTensor(indices)



class GeneradorCiudades(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GeneradorCiudades, self).__init__()
        self.hidden_size = hidden_size

        # Capa RNN: input_size es el tamaño del alfabeto
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Capa de salida: de la memoria a una predicción de letra
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: letra actual. hidden: memoria anterior
        # La RNN nos da la salida y la NUEVA memoria
        out, hidden = self.rnn(x, hidden)

        # Pasamos por la capa lineal para obtener la puntuación de cada letra
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        # Función para resetear la memoria al empezar una palabra nueva
        return torch.zeros(1, 1, self.hidden_size)

# Instanciamos (alfabeto -> entrada/salida, 128 neuronas de memoria)
modelo = GeneradorCiudades(n_letras, 128, n_letras)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# Entrenamiento simple
for epoch in range(1000):
    total_loss = 0
    for ciudad in ciudades:
        # 1. Preparar datos y resetear memoria
        tensor_x = palabra_a_tensor(ciudad)          # Ej: "MADRID"
        tensor_y = palabra_a_objetivo(ciudad)        # Ej: "ADRID."
        hidden = modelo.init_hidden()

        optimizer.zero_grad()
        loss = 0

        # 2. Bucle temporal (aquí es donde la red "recorre" la palabra)
        for i in range(tensor_x.size(0)):
            # Metemos la letra i y la memoria actual
            output, hidden = modelo(tensor_x[i].unsqueeze(0), hidden)

            # Calculamos el error comparando con la letra que debería seguir
            loss += criterion(output.view(1, -1), tensor_y[i].unsqueeze(0))

        # 3. Optimización tras procesar TODA la palabra
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1}/1000 - Error: {total_loss/len(ciudades):.4f}")


def generar_ciudad(letra_inicio):
    modelo.eval()
    with torch.no_grad():
        char = letra_inicio
        nombre = char
        hidden = modelo.init_hidden()

        # Máximo 15 letras para evitar bucles infinitos
        for _ in range(15):
            x = palabra_a_tensor(char)
            output, hidden = modelo(x[0].unsqueeze(0), hidden)

            # Cogemos la letra con mayor probabilidad
            _, top_i = output.topk(1)
            char = int_to_char[top_i.item()]

            if char == '.': break
            nombre += char

    return nombre


def completar_ciudad(prefijo):
    modelo.eval()
    with torch.no_grad():
        hidden = modelo.init_hidden()
        
        # 1. "Calentamos" la memoria con el prefijo
        for i in range(len(prefijo) - 1):
            x = palabra_a_tensor(prefijo[i])
            _, hidden = modelo(x[0].unsqueeze(0), hidden)
        
        # 2. Ahora empezamos a generar desde la última letra del prefijo
        char = prefijo[-1]
        nombre = prefijo
        
        for _ in range(15):
            x = palabra_a_tensor(char)
            output, hidden = modelo(x[0].unsqueeze(0), hidden)
            
            _, top_i = output.topk(1)
            char = int_to_char[top_i.item()]
            
            if char == '.': break
            nombre += char
            
    return nombre



##
# INICIAMOS LA PRUEBA
##
print(generar_ciudad("V")) # Debería intentar escribir MADRID o MALAGA
print(completar_ciudad("MU")) # Debería devolver MURCIA
