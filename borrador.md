


# Borrador

## El problema del Gradiente Desvaneciente (*Vanishing Gradient*)

Para entrenar una red, usamos el **Gradiente Descendente**. El objetivo es
ajustar los pesos () para minimizar el error. Para saber cuánto cambiar un peso
en una capa profunda, usamos la **Regla de la Cadena**. Imagina una red con 4
capas. El ajuste del peso en la primera capa depende de las derivadas de todas
las capas siguientes. Aquí es donde la **función Sigmoide** nos
traicionaba. Mira su derivada:


$$
f'(x) = 0 \quad \text{para } x \neq 0
$$


* La derivada de la sigmoide tiene un valor máximo de **0.25** (cuando la entrada es 0).

* En cuanto la entrada se aleja un poco de cero, la derivada cae a valores cercanos a **0.01** o menos.

Esto ocurre cuando x=0 (el centro de la curva). En el momento en que la neurona está "muy segura" de algo (por ejemplo, el valor de entrada es muy alto, x=10, y la sigmoide devuelve 0.999), la derivada se convierte en:


$$
0.999 \cdot (1-0.999) = 0.999 \cdot 0.001 = 0.000999
$$

La señal de aprendizaje se vuelve mil veces más pequeña solo porque la neurona está en una zona de saturación. Es como un interruptor de luz que tiene un muelle muy fuerte en los extremos. Si intentas moverlo cuando ya está casi arriba o casi abajo, apenas cede. Solo en el centro tiene algo de juego, pero ese "juego" máximo es de apenas un cuarto de la fuerza que le aplicas. Para visualizar esto ejecutar este código:


```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoide(x): # Derivada
    s = sigmoide(x)
    return s * (1 - s)

x = np.linspace(-50, 50, 100)

y1 = sigmoide(x)
y2 = d_sigmoide(x)  

plt.plot(x, y1, label="y = sigmoide")
plt.plot(x, y2, label="y = d_sigmoide")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```

Hay que recordar que el valor del perceptrón suele interpretarse como una probabilidad. Si este tiene que tomar una decisión (si una imagen es perro o gato, por ejemplo), devolviendonos un 0.5 nos estaría indicando que no tiene ni idea de lo que es. Aquí es donde viene el **principal problema de la sigmoide**: Si la neurona está muy segura pero está equivocada (por ejemplo, devuelve un 0.99 de que es un gato, pero la etiqueta real dice que es un perro), necesitamos corregirla con mucha fuerza.




# Proceso de aprendizaje


Aprender significa **actualizar los pesos** para que el error disminuya. La fórmula de actualización para cualquier peso es:

$$
w_{nuevo} = w_{viejo} - (\text{Tasa de aprendizaje} \cdot  \text{Gradiente})
$$

Primeramente tenemos la **tasa de aprendizaje** que es básicamente el nivel de
intensidad con el que queremos que el modelo o neurona aprenda. Imagina que
estás en la montaña, a oscuras, y quieres bajar al valle (el error mínimo). La
tasa de aprendizaje es la longitud de tus pasos. Si la tasa es muy alta das
pasos de gigante. El problema es que podrías pasarte de largo el valle,
aterrizar en la ladera opuesta, y empezar a rebotar de un lado a otro sin llegar
nunca al fondo. El modelo se vuelve inestable y "no converge". Si la tasa es muy
baja das pasos de hormiga. Llegarás al valle con mucha precisión, pero podrías
tardar años (necesitarías millones de iteraciones en el entrenamiento). Además,
podrías quedarte atrapado en un pequeño bache (mínimo local) pensando que ya has
llegado al fondo.

El **Gradiente** es lo que nos dice en qué dirección y con qué fuerza debemos
mover el peso. En una red neuronal, para saber cómo afecta un peso de la Capa 1 al error final
(que se mide en la Salida), tenemos que aplicar la regla de la cadena a través
de todas las capas intermedias. Imagina este flujo simplificado de error:

```
                Error -> Capa 3 -> Capa 2 -> Capa1
```

El gradiente que llega a la Capa 1 es el producto de las derivadas de las capas
superiores. La derivada es la "velocidad" a la que fluye la información del
error. Si la **derivada es alta** (cercana a 1 o mayor): El error viaja con
fuerza. El peso w1​ recibe un mensaje claro: *"¡Oye! Te has equivocado mucho,
cambia tu valor rápido"*. Hay aprendizaje. Si la **derivada es pequeña** (como
el 0.25 máximo de la sigmoide): El error se va "atenuando" en cada capa. Como
vimos, $0.25 \cdot 0.25 \cdot 0.25...$ se convierte en casi nada. Si **la derivada es 0**
(Saturación): El mensaje de error se multiplica por cero. El peso w1​ recibe:
"Cambia tu valor en 0". Es decir, no cambies. No hay aprendizaje.

La derivada es el multiplicador del error. Si el multiplicador es muy pequeño
(como en la sigmoide), la señal de error se desvanece antes de llegar a las
primeras capas de la red. Es como intentar enviar electricidad a través de un
cable con muchísima resistencia: al final del cable no llega voltaje suficiente
para encender la bombilla.

De ahí que se use la función ReLU como función de activación. Esta función tiene
una derivada cercana al 1. El problema surge cuando los valores caen en zona
negativa. En ese caso el aprendizaje se anula porque le gradiente se
vuelve 0. Queda bloqueada porque no recibe ninguna señal de actualización de sus
pesos. Es el equivalente a un circuito con un fusible fundido: la corriente (el
aprendizaje) simplemente deja de pasar.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

x = np.linspace(-50, 50, 100)

y1 = relu(x)
y2 = d_relu(x)  

plt.plot(x, y1, label="y = relu")
plt.plot(x, y2, label="y = d_relu")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```



# El Transformer

En 2017, investigadores de Google publicaron un artículo con un título casi
arrogante: «Attention is All You Need» (La atención es todo lo que
necesitas). Este paper cambió las reglas del juego y jubiló a las arquitecturas
que dominaron la IA durante décadas. Hace años, para procesar texto usábamos
Redes Neuronales Recurrentes (RNN). Imagina que lees una frase palabra por
palabra, de izquierda a derecha, y tratas de mantener el significado en una
«memoria interna».  Si la frase era muy larga, para cuando llegabas al final, la
red ya se había «olvidado» del principio. Además, eran lentas porque no se
podían paralelizar: para procesar la palabra 10, tenías que haber procesado la
1, la 2... hasta la 9. El **Transformer** eliminó la lectura secuencial. En
lugar de leer en orden, mira toda la frase a la vez. ¿Cómo sabe qué palabra es
importante? Mediante la Atención. Observa estas dos frases

* *El banco estaba cerrado porque el financiero llegó tarde.*
* *El banco estaba verde porque lo acababan de pintar.*

La palabra «banco» es la misma, pero su significado cambia. El Transformer usa
el mecanismo de atención para *mirar* las palabras de alrededor: en la primera
frase, «banco» presta atención a «financiero». En la segunda frase, «banco»
presta atención a «pintar».

Esto supuso una revolución principalmente porque como procesamos todas las
palabras a la vez, podemos usar miles de núcleos de GPU simultáneamente. Ya no
hay que esperar a que termine la palabra anterior. Podemos encontrar relaciones
entre una palabra al inicio de un libro y otra al final. Y por último porque la
representación matemática de una palabra cambia según las palabras que tiene al
lado. Imagina que el Transformer usa tres vectores para cada palabra:

* Query (Búsqueda): ¿Qué estoy buscando?
* Key (Clave): ¿Qué información ofrezco?
* Value (Valor): La información que tengo.

Si la palabra es «Programar», su Query podría ser «busco un lenguaje». Si otra
palabra en la frase es «Python», su Key dice «yo soy un lenguaje». Al coincidir,
la atención se dispara. Imagina ahora otro ejemplo: tenemos la frase: «El
sistema operativo dio un error porque el kernel no estaba actualizado». En este
caso, el sujeto omitido «el» (o \emph{it} en inglés) debería hacer referencia (o
prestar más atención) al concepto de Kernel y no de sistema operativo. Para una
IA antigua, si la frase era muy larga, podía perder la conexión entre el
principio («sistema operativo») y el final. El Transformer, gracias al mecanismo
de atención, crea un «enlace de alta velocidad» entre esas palabras, sin
importar la distancia física que las separe en el texto.


Para montar todo este tinglado se usan tres vectores que serán multiplicados
escalarmente.

1. Query (Q): Piensa en esto como una etiqueta de búsqueda. (Ej: "Busco al sujeto de esta acción").
2. Key (K): Piensa en esto como la etiqueta de identificación de cada palabra. (Ej: "Yo soy un sustantivo", "Yo soy un verbo").
3. Value (V): Es la información real de la palabra si resulta que es la elegida.

Para calcular esta «atención» se calcula este producto escalar entre el vector Query de una palabra y los vectores Key de todas las demás. Si el producto escalar es alto, las palabras "encajan" (como una búsqueda en Google con buenos resultados). Se aplica una función Softmax para que los resultados sean probabilidades (que sumen 1). Finalmente, multiplicamos esas probabilidades por los Values.

$$
\mathrm{Attention}(Q, K, V)
=
\mathrm{softmax}\!\left(
\frac{Q K^{\mathsf{T}}}{\sqrt{d_k}}
\right) V
$$

Antes, para entrenar una IA, tenías que pasarle los datos uno a uno (secuencial). Con los Transformers, le lanzas todo el dataset de golpe a la GPU. Esto permitió entrenar modelos con miles de millones de parámetros (los LLMs como GPT-4) usando básicamente toda la información disponible en Internet.


## Un ejemplo real

Vamos a simplificar al máximo para que los números no nos nublen la vista. Imaginemos una "frase" de solo dos palabras: "IA avanza". Antes de las matrices Q,K,V, cada palabra es un vector de números (su "coordenada" de significado). Esto es lo que se llama representación o *embeddings*:


* IA = $[1,0]$
* avanza = $[0,1]$

¿Por qué estas coordenadas? Es una simplificación. Imagina un espacio cartesiano. En la IA clásica de hace 20 años, usábamos representaciones discretas (como índices en un array). Hoy usamos vectores densos. Si estuviéramos en un espacio de solo 2 dimensiones (como el de mi ejemplo [x,y]), los ejes podrían representar conceptos abstractos:

* Eje X: Grado de "Tecnología".
* Eje Y: Grado de "Acción/Movimiento".

En nuestro ejemplo, «IA» $[1,0]$ está muy a la derecha en "Tecnología", pero en 0 en "Acción" y   Avanza $[0,1]$ está en 0 en "Tecnología", pero muy arriba en "Acción".

Para generar Q,K,V, multiplicamos el *embedding* por tres matrices diferentes. Para este ejemplo, supongamos que el modelo ya ha aprendido estos pesos:


$$
W^Q =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}
\quad \text{(para buscar significado)}
$$

$$
W^K =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}
\quad \text{(para ofrecer información)}
$$

$$
W^V =
\begin{pmatrix}
10 & 0 \\
0 & 10
\end{pmatrix}
\quad \text{(el valor real del concepto)}
$$


Multiplicamos el vector de IA $[1,0]$ por las matrices:

* $Q_{IA} = [1,0] \times W^Q = [2,1]$
* $K_{IA} = [1,0] \times W^K = [2,1]$
* $V_{IA} = [1,0] \times W^V = [10,0]$

Y hacemos lo mismo para la palabra "avanza" $[0,1]$:

* $Q_{avanza} = [0,1] \times W^Q = [1,2]$
* $K_{avanza} = [0,1] \times W^K = [1,2]$
* $V_{avanza} = [0,1] \times W^V = [0,10]$


Llega ahora el momento de la verdad. Queremos saber cuánta Atención presta la palabra "IA" a sí misma y a la palabra "avanza". Usamos la fórmula $Q \cdot K^T$ (producto escalar).

1. Atención de «IA» a «IA»:
$$
    [2,1]\cdot[2,1] = (2 \times 2) + (1 \times 1) = {\bf 5}
$$ 

2. Atención de «IA» a «avanza»:
$$
    [2,1]\cdot[1,2] = (2 \times 1) + (1 \times 2) = {\bf 4}
$$ 

El modelo dice: "He sacado un 5 y un 4". Pasamos esto por una función Softmax para que sumen 1  o lo que es lo mismo, 100%:

* Atención a "IA": 0.73 (73%)
* Atención a "avanza": 0.27 (27%)

Así que el resultado final sería un nuevo vector para «IA». Este será la suma ponderada de los valores ($V$):

$$
    Z_{IA} = 0.73 \times V_{IA} + 0.27 \times V_{avanza}  
$$

$$
    Z_{IA} = 0.73 \times [10,0] + 0.27 \times [0,10] = [{\bf 7.3, 2.7} ]
$$

Vamos a resumir todo el proceso. Originalmente, la palabra «IA» era solo
$[1,0]$. Después de la atención, su nuevo vector es $[7.3,2.7]$. ¡Ha absorbido parte
del significado de «avanza»! Ahora el sistema no ve «IA» como un concepto
aislado, sino como una «IA que está avanzando». Si la frase fuera «IA
retrocede», el producto escalar con «retrocede» daría otros números y el vector
final de «IA» sería totalmente distinto.

Recordemos el significado de las coordenadas de significado, donde en este
ejemplo $x$ representaba la cantidad de significado respecto del concepto de
«tecnología» mientras que $y$ representaba la cantidad de significado respecto
de «acción». Cuando el Transformer aplica la Atención, lo que está haciendo
matemáticamente es una combinación lineal de esos vectores. Recordarás de
álgebra que si sumas dos vectores, obtienes un vector resultante que apunta en
una dirección intermedia. En el ejemplo anterior, el nuevo vector de "IA" pasó a
ser $[7.3,2.7]$. Antes, la IA era un concepto estático (x=1, tecnología
pura). Después, el vector se ha "desplazado". Ahora tiene un componente de
y=2.7. Matemáticamente, el modelo ha "movido" la palabra IA hacia la zona de la
"Acción". Si la frase fuera "La IA se detiene", el vector resultante de IA se
movería hacia las coordenadas de "estatismo", porque «estatismo» tendría una
coordenada [0,0] o [0,-1] quizás.


Para el modelo, **el significado es la posición. Si dos vectores están cerca, es
que significan cosas similares en ese contexto**. En una frase sobre medicina,
la palabra "operación" se moverá hacia vectores como "cirugía" o "médico". En
una frase sobre matemáticas, esa misma palabra "operación" se moverá hacia
"suma" o "cálculo". El vector final de una palabra, al final es **el resultado
de mirar a todos sus vecinos y decidir, mediante el producto escalar que vimos,
cuánto de cada vecino debe incorporar a su propia identidad**. Al final de la
capa de atención, ninguna palabra es una isla; todas han "absorbido" un poco de
las demás para formar un significado global de la frase.



# Fundamentos de la IA Generativa

A diferencia de la IA de hace 20 años, donde necesitábamos datos etiquetados por
humanos (esto es un gato, esto es un perro), los modelos actuales usan
Aprendizaje Auto-supervisado. Le damos al modelo billones de páginas de Internet
y le obligamos a jugar a un juego: Predecir el siguiente token (palabra o parte
de palabra). Por ejemplo si la entrada es «el cielo es ...» el modelo puede
predecir que lo que sigue es «verde» donde tendriamos un error alto. El
gradiente entonces debería corregir los pesos hacia atrás, como hemos visto
anteriormente. También podría predecir que sigue «azul» y en ese caso el erro
sería bajo y los pesos se reforzarían. Al final del Transformer, hay una capa
final (una Softmax gigante) que no da una respuesta, sino una distribución de
probabilidad sobre todo el vocabulario (que suele ser de unas 50,000 a 100,000
palabras).
