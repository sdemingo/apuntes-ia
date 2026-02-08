import numpy as np


# X (Entrada): (4,2) — 4 combinaciones de 2 entradas.
# W1  (Pesos Ocultos): (2,2) — Conecta 2 entradas con 2 neuronas ocultas.
# W2  (Pesos Salida): (2,1) — Conecta 2 neuronas ocultas con 1 salida.



# Función de activación: Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada de la sigmoide: necesaria para el Backpropagation
# Si y = sigmoid(x), entonces la derivada es y * (1 - y)
def sigmoid_derivative(y):
    return y * (1 - y)



# Fijamos una semilla para que los resultados sean reproducibles
np.random.seed(42)

# Datos de entrada (XOR) y resultados esperados (Y)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Arquitectura: 2 entradas -> 2 neuronas ocultas -> 1 salida
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Pesos (Weights) y Sesgos (Biases)
# W1 conecta Entrada con Oculta (2x2)
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))

# W2 conecta Oculta con Salida (2x1)
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

learning_rate = 0.5
epochs = 10000

# 4. Bucle de entrenamiento
for epoch in range(epochs):
    # --- FORWARD PASS ---
    # Capa oculta
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Capa de salida
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)

    # --- BACKPROPAGATION ---
    # A. ¿Cuánto nos equivocamos en la salida?
    error = y - predicted_output
    
    # B. Gradiente en la salida (Error * Derivada de la activación)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # C. ¿Cuánto error le corresponde a la capa oculta? (Error retropropagado)
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # --- ACTUALIZACIÓN DE PESOS (Gradiente Descendente) ---
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# 5. Verificación final
print("Resultados después del entrenamiento:")
print(predicted_output)
