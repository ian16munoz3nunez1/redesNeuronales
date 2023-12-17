% Ian Mu;oz Nu;ez - Perceptron
% Ejemplo: Si se quiere entrenar un perceptron con
% una funcion AND de dos entradas, se tendria:
%       +-------------------+
%       | x_1 | x_2 | b | d |
%       +-----+-----+---+---+
%       |  0  |  0  | 1 | 0 | <-- Primer patron
%       +-----+-----+---+---+
%       |  0  |  1  | 1 | 0 | <-- Segundo patron
%       +-----+-----+---+---+
%       |  1  |  0  | 1 | 0 | <-- Tercer patron
%       +-----+-----+---+---+
%       |  1  |  1  | 1 | 1 | <-- Cuarto patron
%       +-----+-----+---+---+
% En donde 'x_1' es la entrada 1, 'x_2', la entrada 2, 'b'
% es la entrada del bias, que siempre está en 1 y 'd' son
% los valores deseados.

close all
clear
clc

x = [0 0 1 1;
    0 1 0 1]; % Datos de entrada
b = [1 1 1 1]; % Entrada fija para el bias
x = [x; b]; % Datos de entrenamiento con entrada fija
d = [0 0 0 1]; % Salida deseada
w = rand(3,1); % Pesos sinapticos aleatorios iniciales
epocas = 100; % Numero de iteraciones deseadas

% Funcion escalon
function y = escalon(v)
    if v >= 0
        y = 1;
    else
        y = 0;
    end
end

p = size(x,2); % Numero de patrones de entrada
% Inicio del entrenamiento
for epoca= 1:epocas
    ep = 0;
    for i= 1:p
        v = w' * x(:,i); % Multiplicacion del vector de entrada por el vector de pesos sinapticos
        y(i) = escalon(v); % Funcion de activacion

        e(i) = d(i) - y(i); % Error obtenido
        if e(i) < 0 || e(i) > 0
            w = w + e(i)*x(:,i); % Ajuste de pesos
            ep = ep + 1;
        end
    end

    % Si no hubo ningun error en el entrenamiento, se termina el proceso
    if ep == 0
        break
    end
end

xl = -0.1; % Limite inferior para mostrar la grafica
xu = 1.1; % Limite superior para mostrar la grafica
t = xl:0.1:xu; % Arreglo de valores para el hiperplano separador

m = -w(1)/w(2); % Pendiente del hiperplano
b = -w(3)/w(2); % Coeficiente del hiperplano
f = m*t + b; % Funcion del hiperplano separador
% f = -(w(1)/w(2))*t - (w(3)/w(2)) % Funcion del hiperplano separador

figure(1)
hold on
grid on
axis equal % Muestra la escala de los ejes igual
axis([xl xu xl xu]) % Limite de los ejes

% Informacion de la grafica
title("and", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)

% Grafica de los datos de entrada y su clasificacion
plot(x(1,y==1), x(2,y==1), 'y*', 'LineWidth', 6, 'MarkerSize', 12)
plot(x(1,y==0), x(2,y==0), 'g*', 'LineWidth', 6, 'MarkerSize', 12)
plot(t, f, 'r-', 'LineWidth', 2) % Grafica del hiperplano separador

