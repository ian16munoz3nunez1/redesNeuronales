% Ian Mu;oz Nu;ez - Adaline
% Desarrollar un código en el que un Adaline sea capaz de aproximar una
% función generada con datos aleatorios utilizando dos tipos de métodos,
% el SGD (Stochastic Gradient Descent) y BDG (Batch Gradient Descent)
% para comparar su funcionamiento
%%%%%%%%%%%%%
%%%% SGD %%%%
%%%%%%%%%%%%%
% Realiza un ajuste de pesos y bias con cada uno de los patrones de
% entrada, al hacer esto, se ajusta en pocas epocas, pero con muchos
% patrones de entrada se vuelve lento y tarda más tiempo en terminar
% el entrenamiento y alcanzar un ajuste óptimo.
%%%%%%%%%%%%%
%%%% BGD %%%%
%%%%%%%%%%%%%
% Este método realiza el ajuste de pesos y bias utilizando todos los
% patrones de entrada, debido a esto, el ajuste optimo tarda más
% epocas en ser alcanzado, pero al haber muchos patrones de entrada
% no afecta mucho al tiempo de entrenamiento ni a los recursos
% computacionales

close all
clear
clc

xl = -0.02; % Limite inferior para generar salidas deseadas aleatorias
xu = 0.02; % Limite superior para generar salidas deseadas aleatorias
n = 100; % Numero de patrones de entrada y salidas deseadas

x = linspace(0,1,n); % Patrones de entrada
d = (1-x) + (xl+(xu-xl)*randn(1,n)); % Salida deseada
eta = 1e-1; % Factor de aprendizaje
epocas = 1000; % Numero de iteraciones

p = size(x,2); % Numero de patrones de entrada
t = 0:0.1:1; % Arreglo de valores para el hiperplano separador
w = -2+(2+2)*rand(); % Peso sinaptico aleatorio entre -2 y 2
b = -2+(2+2)*rand(); % Bias aleatorio entre -2 y 2

pw = []; % Arreglo para guardar el valor de los pesos sinapticos
pb = []; % Arreglo para guardar el valor del bias

for epoca= 1:epocas
    for i= 1:p
        pw = [pw, w]; % Se guarda el valor del peso en el arreglo
        pb = [pb, b]; % Se guarda el valor del bias en el arreglo

        y = w'*x(i) + b; % Interaccion de la entrada con los pesos y el bias

        e = d(i) - y; % Error entre la salida deseada y la obtenida
        w = w + eta*e*x(i); % Ajuste de pesos sinapticos
        b = b + eta*e; % Ajuste del valor del bias

        %% Animacion
        % px = [0,-b/w];
        % py = [b,0];
        % m = (py(2)-py(1))/(px(2)-px(1));
        % f = m*(t-px(1)) + py(1);
        % cla
        % hold on
        % grid on
        % title(epoca, 'FontSize', 20)
        % xlabel('x', 'FontSize', 15)
        % ylabel('y', 'FontSize', 15)
        % plot(x, d, 'b*', 'LineWidth', 2)
        % plot(t, f, 'r-', 'LineWidth', 2)
        % drawnow
    end

    J(epoca) = sum((d'-(x'*w+b)).^2); %  Se guarda el error del Adaline
end

px = [0,-b/w]; % Posiciones en 'x' para los puntos que definen al hiperplano
py = [b,0]; % Posiciones en 'y' para los puntos que definen al hiperplano
m = (py(2)-py(1))/(px(2)-px(1)); % Pendiente del hiperplano
f = m*(t-px(1)) + py(1); % Funcion del hiperplano

figure(1)

subplot(2,2,1) % Grafica de los patrones de entrada, salida deseada y el hiperplano
hold on
grid on
title("Metodo SGD", 'FontSize', 20)
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

