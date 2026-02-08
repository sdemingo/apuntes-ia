---
geometry: margin=2cm
numbersections: true
title: "Apuntes sobre la IA"
author: "Sergio de Mingo"
date: "2 de febrero de 2026"
---


Estos apuntes son parte de la conclusiones sacadas a partir del estudio con
Gemini de los conceptos aquí desarrollados.


# El perceptron


El Perceptrón (propuesto por Rosenblatt en los 50) es la unidad atómica de la
IA. Es básicamente una función matemática que intenta imitar a una neurona
biológica. No es más que un clasificador lineal binario. Un clasificador binario
es una función que puede decidir si una entrada, representada por un vector de
números, pertenece o no a una clase específica.

$$
y = f\!\left( \sum_{i=1}^{n} w_i x_i + b \right)
$$

* Entradas (*x*): Los datos que llegan (ej. valores de píxeles).

* Pesos (*w*): La "importancia" que la red le da a cada entrada. Aprender en IA es, literalmente, ajustar estos números.

* Bias (*b*): El sesgo, que permite desplazar la función de activación hacia la izquierda o derecha para ajustarse mejor a los datos.

* Función de activación (*f*): Decide si la neurona "dispara" una señal o no. Antiguamente usábamos la Heaviside (escalón) o la Sigmoide.


La limitación de este algoritmo es que si dibujamos en un gráfico estos
elementos, se deben poder separar con un hiperplano únicamente los elementos
«deseados» discriminándolos (separándolos) de los «no deseados». Esto lo veremos
mejor en el punto \ref*{multicapa}, donde estudiaremos una Red Neuronal Clásica
(Multilayer Perceptron). Este tipo no es más que muchas de estas neuronas
organizadas en capas: una de entrada, una o varias ocultas y una de salida.


## El aprendizaje de un perceptrón

El proceso de aprendizaje de un perceptrón comienza con una fase de
**inicialización**, donde el modelo parte de un estado de desconocimiento
absoluto, asignando valores aleatorios a sus pesos y un valor inicial
(normalmente cero) a su sesgo o bias. Una vez configurado, **el aprendizaje se
convierte en un ciclo iterativo que se repite durante varias épocas**. En cada
iteración, el perceptrón realiza primero un forward pass o predicción: toma los
datos de entrada, los multiplica por sus respectivos pesos, suma el sesgo y pasa
el resultado por una función de activación (como la función escalón) para
decidir si la neurona debe activarse o no. Inmediatamente después, el modelo
compara su predicción con el valor real ($y_i$) que debería haber obtenido,
calculando así el error:

$$
    Error = y_i - f(w_i \cdot x_i + b)
$$


Si existe una discrepancia, entra en juego la regla de aprendizaje, donde el
perceptrón ajusta sus pesos internos de forma proporcional a tres factores: la
magnitud del error cometido, el valor de la entrada que causó dicho error y,
fundamentalmente, la tasa de aprendizaje, que actúa como un regulador de la
velocidad del cambio. 



$$
    w_i = w_i + (x_i \times \text{Error} \times \text{Tasa de aprendizaje})
$$



Este ajuste busca "mover" la frontera de decisión del perceptrón (la línea recta
que separa las clases) para que, en el siguiente intento, la predicción sea más
precisa. A través de la repetición constante de este proceso con todos los datos
del conjunto de entrenamiento, los pesos convergen gradualmente hacia unos
valores óptimos que permiten al modelo clasificar correctamente las entradas,
logrando así que la máquina "aprenda" la lógica subyacente de los datos,


## Trabajando con el perceptrón

Para empezar a programar un perceptrón necesitamos establecer su estado interno,
tanto los pesos $W$ como el sesgo $b$. Normalmente estos se inicializarán con
valores pequeños. Tras esto haremos el cálculo de la predicción en sí, donde
aplicamos la fórmula inicial del perceptrón con la aplicación de la función de
activación. Por último, el aprendizaje o el ajuste de los pesos. En este caso
esto se ven el método `init()`. Este método recibe además el número de inputs
que tendrá el perceptrón para poder generar un array de pesos adecuado. El
método `predict()` es básicamente la aplicación de la función de activación para
cada componente del vector de entrada `X`. En este caso `Z` es el producto de
cada componente del vector por el peso asignado (y al que al final sumamos el
sesgo). Como activación usamos la función escalón por lo que el resultado será 1
si $z > 0$ y 0 en cualquier otro caso. El mecanismo fundamental se encuentra en
el método `train()`. En este método es donde se irá probando al perceptrón y
ajustando sus pesos hasta que el resultado sea el esperado. En este caso
probamos 100 veces. En cada prueba realizamos el proceso de
aprendizaje/reajuste, teniendo en cuenta que los resultados esperados los
tenemos en el vector `y_AND`:

1. Calculamos la predicción para esa entrada: `prediction = self.predict(X[i])`
2. Calculamos el error comparando el resultado con lo esperado: `error = y[i] - prediction`
3. Recalculamos los pesos: `self.weights += error * self.lr * X[i]`
4. Recalculamos el sesgo: `self.bias += error * self.lr`


```
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z > 0 else 0

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += error * self.lr * X[i]
                self.bias += error * self.lr


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y_AND = np.array([0, 0, 0, 1])
y_OR = np.array([0, 1, 1, 1])

p = Perceptron(n_inputs=2)
p.train(X, y_AND)

print("Probamos que sepa calcular una operación AND u OR:")
for t in X:
    print(f"Entrada: {t} -> Predicción: {p.predict(t)}")
```

La magia se produce cuando tras probar el perceptrón con el vector de
aprendizaje `y_AND` y viendo que ha sabido hacer una operación AND cambiamos el
vector por `y_OR` y aprende a hacer una OR haciéndola correctamente para las
mismas entradas.

Siguiendo este razonamiento nos topamos con el primer gran problema del
perceptrón. Al intentar implementar este mismo aprendizaje para XOR usando un
vector `y_XOR = np.array([0, 1, 1, 0])` vemos que es imposible. Si visualizamos
el espacio de entrada como un plano con dos ejes y cuatro puntos
(0,0),(0,1),(1,0),(1,1), vemos que en el AND, solo el punto (1,1) es
positivo. Puedes dibujar una línea recta que separe ese punto de los otros
tres. Lo mismo pasa en el OR donde tres puntos son positivos y solo el (0,0) es
negativo. También puedes trazar una línea recta para separarlos.



# Redes multicapa
\label{multicapa}

Si recordamos el razonamiento del punto anterior donde definíamos en un plano de
coordenadas los resultados del perceptrón, si intentamos separar los resultados
positivos del XOR vemos que es imposible hacerlo con una sola línea. En el XOR,
los puntos positivos (los resultados válidos con los que se obtiene un 1) son
(0,1) y (1,0) y los negativos son (0,0) y (1,1). Es imposible separar ambos
espacios (de positivos y negativos) con una sola línea recta. Necesitas dos o
bien una curva. Esto se explica matemáticamente porque un perceptrón simple es
matemáticamente un hiperplano. En 2D, es una recta. Si tus datos no son
"linealmente separables", el perceptrón se quedará oscilando para siempre sin
encontrar una solución. Para resolver el XOR, necesitamos añadir hacer una
pequeña red multicapa:

1. **Capa de entrada** formada por dos neuronas: Recibirá los valores $x_1$ y $x_2$
2. **Capa oculta** formada por dos o tres neuronas: Aquí es donde ocurre la magia debido a que estas neuronas crearán nuevas dimensiones.
3. **Capa de salida** formada por una neurona que nos da el resultado final de la operación: 0 o 1.


Para realizar el cálculo ya no multiplicaremos el peso por la entrada en bucle,
como haciamos antes. Ahora usaremos multiplicación de matrices. Siendo $X$ la
matriz de entrada y $W_1$ la matriz de pesos de la capa oculta calcularemos $Z_1
= X \cdot W_1 + b_1$ y luego aplicaremos la función de activación. En este caso
usaremos la sigmoide. Para la capa de salida calcularemos $Z_2 = A_1 \cdot W_2 +
b_2$ y de nuevo activaremos con la función sigmoide. Como nota indicamos que
usamos la función sigmoide debido a que es una función no lineal. Si usáramos
una función lineal, por muchas capas que pongamos, la red seguiría siendo una
simple combinación lineal (una sola línea recta). La no-linealidad es lo que
permite "curvar" el espacio necesario para este caso.

## Aprendizaje en varias capas

El concepto que revolucionó la IA en los 80 fue: ¿Cómo el error se calcula al
final, en la salida? ¿cómo sabemos cuánto deben cambiar los pesos de la primera
capa? El proceso de aprendizaje ahora se basa en estos tres pasos:

1. El *Forward Pass* o la multiplicación de las matrices y la activación
2. El cálculo del error; ¿Cuánto nos queda para el valor real?
3. El *Backpropagation* donde calculamos el «gradiente» de cada capa usando La
   Regla de la cadena.
4. Actualizamos los pesos restando el gradiente.


El **Gradiente** es lo que nos dice en qué dirección y con qué fuerza debemos
mover el peso para que el error total disminuya lo más rápido posible. 
En una red neuronal, para saber cómo afecta un peso de la Capa 1
al error final (que se mide en la Salida), tenemos que aplicar **La Regla de La
Cadena** a través de todas las capas intermedias. El gradiente que llega a la
Capa 1 es el producto de las derivadas de las capas superiores. La derivada es
como la velocidad a la que fluye la información del error. Si la derivada es
alta (cercana a 1 o mayor): El error viaja con fuerza. El peso $w_1$ recibe un
mensaje claro: *"¡Oye! Te has equivocado mucho, cambia tu valor rápido"*. Hay
aprendizaje. Si la derivada es pequeña, el error se va atenuando en cada
capa. ¿Por qué usamos derivadas? El objetivo de la red es minimizar una función
de Error (o Pérdida). Imagina que el error es una montaña y tú estás en la cima,
a ciegas. Quieres bajar al valle (error cero). ¿Cómo sabes hacia dónde dar el
paso? Tocando el suelo con el pie para ver la pendiente. Esa pendiente es la
derivada. Si la derivada es positiva, el terreno sube; vas hacia atrás. Si es
negativa, el terreno baja; vas hacia adelante. El uso de la regla de la cadena
es una cuestión de esto anterior. En definitiva una red neuronal es una función
compuesta gigante. Si tenemos dos capas, la salida (Y) es:

$$
Y = \mathrm{Activación}_2 \left(
W_2 \cdot \mathrm{Activación}_1 \left(
W_1 \cdot X
\right)
\right)
$$

Cuando queremos saber cómo afecta el peso de la primera capa $W_1$ al error
final, tenemos que "deshacer" la función de fuera hacia adentro. La Regla de la
Cadena nos dice que **la derivada de una función compuesta es el producto de las
derivadas de sus componentes**. Usamos la derivada de la activación porque es la
única forma matemática de saber cuánta responsabilidad tiene una neurona
específica en el error final. Esto es exactamente lo mismo que el análisis de
sensibilidad en ingeniería de sistemas. ¿Cómo afecta una pequeña variación en la
entrada a la salida del sistema? La respuesta es siempre la derivada.

A continuación se muestra el bucle de aprendizaje de la multicapa implementada
de forma completa en `src/multicapa.py`:

```python
for epoch in range(epochs):
    # --- FORWARD PASS ---
    # Capa oculta
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Capa de salida
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)

    # --- BACKPROPAGATION ---
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # --- ACTUALIZACIÓN DE PESOS (Gradiente Descendente) ---
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

```


Vamos a ver esto de forma algo más formal. La regla general para actualizar
cualquier peso en la red es:

$$w_{nuevo} = w_{actual} - \eta \cdot \frac{\partial E}{\partial w}$$

Donde:

* $\eta$ es el learning rate (la longitud del paso).
* $\frac{\partial E}{\partial w}$ es el gradiente, que nos indica la dirección de máxima subida del error. Como queremos bajar, restamos este valor (o lo sumamos si el gradiente ya incluye el signo del error, como en nuestro código).

Para **actualizar los pesos de la salida** $W_2$ debemos calcular el gradiente
para la matriz de pesos completa: 

$$\Delta W_2 = \eta \cdot (A_1^T \cdot
\delta_2)$$ 

Donde $A_1$ es la salida de la capa oculta y $\delta_2$ es el error local de la
salida. Usamos la transpuesta (T) porque queremos relacionar cada neurona oculta
con cada error de salida, creando una matriz de ajustes que coincida con las
dimensiones de $W_2$.


Para **actualizar los pesos de la entrada** $W_1$ debemos aplicar la misma lógica
pero un paso más atrás en la red. Aquí, el gradiente depende de la entrada
original $X$ y del error que hemos "retropropagado" hasta la capa oculta desde
el final.

$$\Delta W_1 = \eta \cdot (X^T \cdot \delta_1)$$













