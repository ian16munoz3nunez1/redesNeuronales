# Redes Neuronales

## Funciones de activación

### Función ***Escalón***

![](.src/escalon.png)

$
\large
\varphi(v) =
\left\{
\begin{matrix}
    1 & \text{si } v \geq 0 \\
    0 & \text{si } v < 0
\end{matrix}
\right.
$

### Función ***Lineal a Tramos***

![](.src/linealATramos.png)

$
\large
\varphi(v) =
\left\{
\begin{matrix}
    1 & v \geq 1 \\
    v & 0 < v < 1 \\
    0 & v \leq 0
\end{matrix}
\right.
$

### Función ***Logística***

![](.src/logistica.png)

$
\large
\varphi(v) = \dfrac{1}{1 + e^{-av}}
$

### Función ***Signo***

![](.src/signo.png)

$
\large
\varphi(v) =
\left\{
\begin{matrix}
    1 & \text{si } v > 0 \\
    0 & \text{si } v = 0 \\
    -1 & \text{si } v < 0
\end{matrix}
\right.
$

### Función ***Tangente Hiperbólica***

![](.src/tanH.png)

$
\large
\varphi(v) = \tanh(v)
$

### Función ***Lineal***

![](.src/lineal.png)

$
\large
\varphi(v) = Av
$

### Función ***Gaussiana***

![](.src/gaussiana.png)

$
\large
\varphi(v) = Ae^{-Bv^2}
$

### Función ***Sinusoidal***

![](.src/sinusoidal.png)

$
\large
\varphi(v) = A\sin(Bv + C)
$

