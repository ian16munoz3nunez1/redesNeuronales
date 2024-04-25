# Redes Neuronales

## Funciones de activación y su derivada

### Función ***Escalón***

![](.src/funcionesActivacion/escalon.png)

$$
\large
y = \varphi(v) =
\left\lbrace
\begin{matrix}
    1 & \text{si } v \geq 0 \\
    0 & \text{si } v < 0
\end{matrix}
\right.
\quad\quad
\varphi(v) \in \lbrace 0, 1 \rbrace
\quad\quad
\nexists \frac{d}{dv} y
$$

No muy utilizada actualmente.

### Función ***Lineal a Tramos***

![](.src/funcionesActivacion/linealATramos.png)

$$
\large
y = \varphi(v) =
\left\lbrace
\begin{matrix}
    1 & v \geq 1 \\
    v & 0 < v < 1 \\
    0 & v \leq 0
\end{matrix}
\right.
$$

### Función ***Logística***

![](.src/funcionesActivacion/logistica.png)

$$
\large
y = \varphi(v) = \dfrac{1}{1 + e^{-av}}
\quad\quad
\varphi(v) \in \big( 0, 1 \big)
\quad\quad
\frac{d}{dv}y = ay(1 - y)
$$

$$
\varphi(v) = \frac{1}{1+e^{-av}} = \left( 1 + e^{-av} \right)^{-1}
$$

$$
\varphi'(v) = (-1)(1+e^{-av})^{-2}(e^{-av})(-a) = \frac{ae^{-av}}{(1+e^{-av})^2}
$$

$$
\varphi'(v) = \left( \frac{a}{1+e^{-av}} \right) \left( \frac{e^{-av}}{1+e^{-av}} \right)
= \left( \frac{a}{1+e^{-av}} \right) \left( \frac{1+e^{-av}-1}{1+e^{-av}} \right)
$$

$$
\varphi'(v) = \left( \frac{a}{1+e^{-av}} \right)
\left( \cancel{\frac{1+e^{-av}}{1+e^{-av}}} - \frac{1}{1+e^{-av}} \right)
$$

$$
\varphi'(v) = a\left( \frac{1}{1+e^{-av}} \right) \left( 1 - \frac{1}{1+e^{-av}} \right)
= a\varphi(v) \left[ 1 - \varphi(v) \right]
$$

Se utiliza en cualquier capa pero sobretodo en la capa de salida para clasificación.

### Función ***Signo***

![](.src/funcionesActivacion/signo.png)

$$
\large
\varphi(v) =
\left\lbrace
\begin{matrix}
    1 & \text{si } v > 0 \\
    0 & \text{si } v = 0 \\
    -1 & \text{si } v < 0
\end{matrix}
\right.
$$

No muy utilizada actualmente.

### Función ***Tangente Hiperbólica***

![](.src/funcionesActivacion/tanH.png)

$$
\large
y = \varphi(v) = a\tanh(bv)
\quad\quad
\varphi(v) \in \big( -1, 1 \big)
\quad\quad
\frac{d}{dv}y = \frac{b}{a}(a-y)(a+y)
$$

$$
\varphi'(v) = ab\text{ sech}^2(bv) = ab(1-\tanh^2(bv)) = ab - ab\tanh^2(bv)
$$

$$
\varphi'(v) = \left( \frac{a}{a} \right) (ab-ab\tanh^2(bv))
= \frac{a^2b - a^2b\tanh^2(bv)}{a}
$$

$$
\varphi'(v) = \frac{b(a^2 - a^2\tanh^2(bv))}{a}
= \left( \frac{b}{a} \right) (a^2 - a^2\tanh^2(bv))
$$

$$
\varphi'(v) = \left( \frac{b}{a} \right) (a - a\tanh(bv)) (a+a\tanh(bv))
= \left( \frac{b}{a} \right) (a-y) (a+y)
$$

Se utiliza solo en capas ocultas.

### Función ***Lineal*** o ***Identidad***

![](.src/funcionesActivacion/lineal.png)

$$
\large
y = \varphi(v) = Av
\quad\quad
\varphi(v) \in (-\infty, \infty)
\quad\quad
\frac{d}{dv}y = \frac{d}{dv}\varphi(v) = A
$$

Se utiliza en la última capa para casos de regresión.

### Función ***Gaussiana***

![](.src/funcionesActivacion/gaussiana.png)

$$
\large
y = \varphi(v) = Ae^{-Bv^2}
\quad\quad
\varphi(v) \in ( 0, 1 ]
\quad\quad
\frac{d}{dv}y = -2Bvy
$$

$$
\large
\varphi'(v) = -2Bv Ae^{-Bv^2}
$$

Se utiliza en arquitecturas específicas como en la ***RBF-NN***.

### Función ***Sinusoidal***

![](.src/funcionesActivacion/sinusoidal.png)

$$
\large
\varphi(v) = A\sin(Bv + C)
$$

$$
\large
\varphi'(v) = A\cos(Bv+C)
$$

### Función ***Softplus***

![](.src/funcionesActivacion/softplus.png)

$$
\large
y = \varphi(v) = \ln(1+e^{av})
\quad\quad
\varphi(v) \in ( 0, \infty )
\quad\quad
\frac{d}{dv}y = \frac{1}{1+e^{-av}}
$$

$$
\varphi'(v) = \frac{ae^{av}}{1+e^{av}}
= a\left[ \frac{e^{av}}{1+e^{av}} \left( \frac{\frac{1}{e^{av}}}{\frac{1}{e^{av}}} \right) \right]
= a\left( \frac{1}{1+e^{-av}} \right)
$$

$$
\varphi'(v) = \frac{a}{1+e^{-av}}
$$

Se utiliza solo en capas ocultas y especialmente en redes profundas.

### Función ***ReLU (Rectified Linear Unit)***

![](.src/funcionesActivacion/relu.png)

$$
\large
y = \varphi(v) = \max(0, v)
\quad\quad
\varphi(v) \in [0, \infty)
\quad\quad
\frac{d}{dv}y =\left\lbrace
\begin{matrix}
    1 & \text{si } v \geq 0 \\
    0 & \text{si } v < 0
\end{matrix}
\right.
$$

Se utiliza solo en capas ocultas y especialmente en redes profundas.

### Función ***Leaky ReLU***

$$
\large
y = \varphi(v) = \left\lbrace
\begin{matrix}
    0.1v & \text{si } v < 0 \\
    v & \text{si } v \geq 0
\end{matrix}
\right.
\quad\quad
\varphi(v) \in (-\infty,\infty)
\quad\quad
\frac{d}{dv}y = \left\lbrace
\begin{matrix}
    0.1 & \text{si } v < 0 \\
    1 & \text{si } v \geq 0
\end{matrix}
\right.
$$

Se utiliza solo en capas ocultas y especialmente en redes profundas. Corrige
algunos detalles de la ***ReLU***.

## La Neurona Artificial

La neurona artificial se representa de la siguiente manera

![](.src/neuron/neuron.png)

Con $n$ entradas $x$, $n$ pesos sinápticos representados con $w$ y un bias $b$,
la acumulación de todas estás señales representa un proceso sumatorio que da
como resultado el valor $v$ que se encuentra debajo.

$$
v = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

$$
v = \sum_{i=1}^p \left[ w_i x_i \right] + b
$$

Representando las ecuaciones anteriores de forma vectorial se tiene lo
siguiente:

$$
x =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
\end{bmatrix}
\quad\quad
w =
\begin{bmatrix}
    w_1 \\
    w_2 \\
    \vdots \\
    w_n
\end{bmatrix}
$$

$$
v = w^\top x + b
$$

Al final de la acumulatoria, se tiene un umbral o función de activación, con lo
que se condiciona la salida de la neurona.

$$
y = \varphi (v)
$$

## El Perceptrón

En el perceptrón, se tiene la misma estructura de la red neuronal.

$$
v = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

$$
v = \sum_{i=1}^p \left[ w_ix_i \right] + b
$$

Con

$$
x =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
\end{bmatrix}
\quad\quad\text{y}\quad\quad
w =
\begin{bmatrix}
    w_1 \\
    w_2 \\
    \vdots \\
    w_n
\end{bmatrix}
$$

$$
v = w^\top x + b
$$

En esta neurona se usa la función de activación de tipo escalón.

$$
y = \varphi (v) =
\left\lbrace
\begin{matrix}
    1 & \text{si } v \geq 0 \\
    0 & \text{si } v < 0
\end{matrix}
\right.
$$

### El Perceptrón en 2 dimensiones

Para clasificar y dibujar un hiperplano separador en un espacio de 2
dimensiones se tendría una representación del perceptrón como la siguiente.

![](.src/perceptron/perceptron0.png)

Así, se tendría el valor de $v$ como

$$
v = w_1x_1 + w_2x_2 + b
$$

Ya que la ecuación anterior se comporta como la ecuación de la recta

$$
0 = ax + by + c
$$

funciona para dibujar un hiperplano separador entre los patrones de entrada,
como en el ejemplo siguiente

![](.src/perceptron0.png)

para dibujar el hiperplano, se despeja $x_2$ de la ecuación

$$
v = w_1x_1 + w_2x_2 + b
$$

resultando en

$$
x_2 = -\dfrac{w_1}{w_2}x_1 - \dfrac{b}{w_2} 
$$

Con $x_1$ siendo una variable que depende del tiempo.
Observando la ecuación anterior, se parece mucho a la función

$$
y = mx + b
$$

Que es otra manera de representar una recta, con esto, se tiene el hiperplano
que muestra la forma en que el perceptrón clasificó los datos de entrada.

### El Perceptrón en 3 dimensiones

Para un caso de 3 dimensiones, se tiene la siguiente ecuación

$$
x_3 = -\dfrac{w_1x_1}{w_3} - \dfrac{w_2x_2}{w_3} - \dfrac{b}{w_3}
$$

Con la representación del perceptrón de la siguiente manera

![](.src/perceptron/perceptron1.png)

Un ejemplo de un perceptrón usado en un caso con datos de entrada de 3
dimensiones

![](.src/perceptron/perceptron1x.png)

### Algoritmo del Perceptrón

```
for epoch in {1, 2, ..., epochs}
    for i in {1, 2, ..., p}
        v = w*x[i] + b
        y = phi(v)
        e = d[i] - y
        
        if e != 0
            w <- w + eta*e*x[i]
            b <- b + eta*e
```

En donde `epochs` es el número de epocas deseadas, `epoch` es la epoca actual,
`p` es el número de datos de entrada, `d` es la salida deseada y `e` la
diferencia entre la salida deseada y la obtenida, por último, `eta` es el
factor de aprendizaje del perceptrón.

### Ejercicios del Perceptrón

[Compuerta AND](https://github.com/ian16munoz3nunez1/redesNeuronales/tree/8f2e6eb6cb54249cb43ad67647a75e61515f2d84)

![](.src/perceptron/perceptron0x.png)

[Resolver A(B+C)](https://github.com/ian16munoz3nunez1/redesNeuronales/tree/7680adffbc0ac76376cf402cfb098e3d4e0e149b)

![](.src/perceptron/perceptron1x.png)

[Clasificación de 4 grupos](https://github.com/ian16munoz3nunez1/redesNeuronales/tree/e455500d96bbbdc9bc1fd6e7c599eb457e330739)

![](.src/perceptron/perceptron2x.png)

## Anexo

### Perceptrón

$$
v = \sum_{i=1}^p \left[ w_ix_i \right] + b
$$

$$
v = w^\top x + b
$$

$$
y = \varphi (v)
$$

$$
e = d_i - y
$$

$$
w \leftarrow w + \eta e x_i
$$

$$
b \leftarrow b + \eta e
$$

$$
x_2 = -\dfrac{w_1}{w_2}x_1 - \dfrac{b}{w_2} 
$$

$$
x_3 = -\dfrac{w_1x_1}{w_3} - \dfrac{w_2x_2}{w_3} - \dfrac{b}{w_3}
$$


