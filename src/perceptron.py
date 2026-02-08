import numpy as np

# Ejemplo donde tenemos un perceptrón que va a aprender 
# el funcionamiento de una puerta AND sencilla. Básicamente
# aprenderá que sin las dos entradas son 1 el resultado es un
# 1 también


class Perceptron:

    # Inicializamos el array de pesos de forma aleatoria 
    # y bias en cero
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0
        self.lr = learning_rate

    # Como función de activación usamos la función escalón
    # donde f(z) = 1 si z>0 y f(z) = 0 en caso contrario
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z > 0 else 0

    # El aprendizaje consiste en 100 pruebas. En cada prueba
    # recorremos el vector de entradas X y le pasamos la entrada 
    # a la función de activación (predict()) y comparamos esto 
    # con el resultado esperado que es el vector Y.
    
    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                # El error es (Valor Real - Valor Predicho)
                error = y[i] - prediction
                
                # REGLA DE APRENDIZAJE:
                # w = w + error * entrada * tasa_aprendizaje
                self.weights += error * self.lr * X[i]
                self.bias += error * self.lr


# Datos de entrenamiento para una puerta AND y para una OR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y_AND = np.array([0, 0, 0, 1])
y_OR = np.array([0, 1, 1, 1])

# Crear y entrenar
print ("Creamos y entrenamos al perceptrón ...")
p = Perceptron(n_inputs=2)
p.train(X, y_OR)

# Probar
print("Probamos que sepa calcular una operación AND:")
for t in X:
    print(f"Entrada: {t} -> Predicción: {p.predict(t)}")
