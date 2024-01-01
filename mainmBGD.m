% Ian Mu;oz Nu;ez - Adaline
% Desarrollar un código en el que un Adaline sea capaz de aproximar una
% función generada con datos aleatorios utilizando ahora el método
% mBGD (mini-Batch Gradient Descent), esta generaliza los metodos SGD
% y BGD, pues dependiendo el tamaño de los mini-lotes que se elige, la
% neurona se puede entrenar con un algoritmo parecido a un SGD o BGD
% (si se selecciona un tamaño de lote de 1, el algoritmo se comporta
% como SGD, si se elige un tamaño de lote del número de patrones de
% entrada, el algoritmo funcionara como un BGD.


close all
clear
clc

xl = -0.02;
xu = 0.02;
n = 100;

x = linspace(0, 1, n); % Datos de entrada
d = (1-x) + (xl+(xu-xl)*randn(1,n)); % Salida deseada
eta = 1e-1; % Factor de aprendizaje
epocas = 1000; % Numero de iteraciones deseadas
batch_size = n; % Tamano de los lotes

p = size(x, 2); % Numero de patrones de entrada
t = 0:0.1:1; % Arreglo de valores para el hiperplano separador
w = -2+(2+2)*rand(); % Pesos sinapticos
b = -2+(2+2)*rand(); % Bias

pw = []; % Arreglo para guardar el valor de los pesos sinapticos
pb = []; % Arreglo para guardar el valor del bias

for epoca= 1:epocas
    xl = 1; % Indice inferior para dividir por lotes
    xu = batch_size; % Indice superior para dividir por lotes
    while xl < p
        pw = [pw, w]; % Se guarda el valor del peso en el arreglo
        pb = [pb, b]; % Se guarda el valor del bias en el arreglo

        mx = x(xl:xu); % Lote de los patrones de entrada
        my = d(xl:xu); % Lote de la salida deseada
        y = w'*mx + b; % Interaccion de la entrada con los pesos y el bias

        e = my - y; % Error entre la salida deseada y la obtenida
        w = w + (eta/p)*e*mx'; % Ajuste de pesos sinapticos
        b = b + (eta/p)*sum(e); % Ajuste del valor del bias

        xl = xl+batch_size; % Incremento del indice inferior para dividir por lotes
        xu = xu+batch_size; % Incremento del indice superior para dividir por lotes
    end

    J(epoca) = sum((d - (w'*x+b)).^2); % Error cuadratico medio
end

px = [0 -b/w]; % Posiciones en 'x' para los puntos que definen al hiperplano
py = [b 0]; % Posiciones en 'y' para los puntos que definen al hiperplano
m = (py(2)-py(1))/(px(2)-px(1)); % Pendiente del hiperplano
f = m*(t-px(1)) + py(1); % Funcion del hiperplano

figure(1)

subplot(2,2,1) % Grafica de los patrones de entrada, salida deseada y el hiperplano
hold on
grid on
title("Metodo mBGD", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
plot(x, d, 'b*', 'LineWidth', 2) % Patrones de entrada y salida deseada
plot(t, f, 'r-', 'LineWidth', 2) % Hiperplano separador

subplot(2,2,2) % Grafica de los valores de pesos sinapticos y bias
hold on
grid on
title("Valor de pesos y bias", 'FontSize', 20)
xlabel('w', 'FontSize', 15)
ylabel('b', 'FontSize', 15)
plot(pw, pb, 'r-', 'LineWidth', 2) % Posiciones de los valores de pesos sinapticos y bias
% Posicion final de los pesos sinapticos y bias
plot(pw(end), pb(end), 'bo', 'MarkerSize', 8, 'LineWidth', 2)
plot(pw(end), pb(end), 'rx', 'MarkerSize', 8, 'LineWidth', 2)

subplot(2,2,3:4) % Grafica del error del Adaline
hold on
grid on
title("Grafica del error", 'FontSize', 20)
xlabel('Epocas', 'FontSize', 15)
ylabel('Error', 'FontSize', 15)
plot(J, 'g-', 'LineWidth', 2) % Error del Adaline

