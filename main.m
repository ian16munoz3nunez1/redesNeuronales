% Ian Mu;oz Nu;ez - Adaline
% Desarrollar un código en el que un Adaline sea capaz de aproximar la función
% descrita por los puntos
% - (1.0, 0.5)
% - (1.5, 1.1)
% - (3.0, 3.0)
% - (-1.2, -1.0)
% Los datos para el entrenamiento del Adaline resultan de la siguiente manera
%       +-----------------+
%       |   x  | b |   y  |
%       +------+---+------+
%       |  1.0 | 1 |  0.5 |
%       +------+---+------+
%       |  1.5 | 1 |  1.1 |
%       +------+---+------+
%       |  3.0 | 1 |  3.0 |
%       +------+---+------+
%       | -1.2 | 1 | -1.0 |
%       +-----------------+
% con las posiciones en 'x' como los datos de entrada y las posiciones en 'y'
% como la salida deseada, y 'b' como la entrada fija del bias.

close all
clear
clc

x = [1.0 1.5 3.0 -1.2]; % Datos de entrada
b = [1 1 1 1]; % Entrada fija para el bias
x = [x; b]; % Datos de entrenamiento con entrada fija
d = [0.5 1.1 3.0 -1.0]; % Salida deseada
w = rand(2, 1); % Pesos sinapticos aleatorios iniciales
epocas = 100; % Numero de iteraciones deseadas
eta = 0.1; % Factor de aprendizaje

%% Entrenamiento
p = size(x, 2); % Numero de patrones de entrada
for epoca= 1:epocas
    for i= 1:p
        y(i) = w' * x(:, i); % Multiplicacion del vector de entrada por el vector de pesos sinapticos

        e(i) = d(i) - y(i); % Error obtenido
        if e(i) < 0 || e(i) > 0
            w = w + eta*e(i)*x(:,i); % Ajuste de pesos
        end
    end
    J(epoca) = sum((d' - x'*w).^2); % Error minimo cuadrado
end

%% Prediccion
for i= 1:size(x,2)
    y(i) = w' * x(:,i);
end

xl = -1.5; % Limite inferior para mostrar la grafica
xu = 3.5; % Limite superior para mostrar la grafica
t = xl:0.1:xu; % Arreglo de valores para el hiperplano separador
xp = [0 -w(2)/w(1)]; % Posiciones en 'x' para los puntos que definen al hiperplano
yp = [w(2) 0]; % Posiciones en 'y' para los puntos que definen al hiperplano
m = (yp(2)-yp(1))/(xp(2)-xp(1)); % Pendiente del hiperplano
f = m*(t-xp(1)) + yp(1); % Funcion del hiperplano

figure(1)
hold on
grid on
axis equal % Muestra la escala de los ejes igual
axis([xl xu xl xu]) % Limite de los ejes

plot(x(1,:), d, 'b*', 'LineWidth', 4, 'MarkerSize', 8) % Datos de entrada y salida deseada
plot(x(1,:), y, 'y*', 'LineWidth', 4, 'MarkerSize', 8) % Prediccion
plot(xp, yp, 'g*', 'LineWidth', 4, 'MarkerSize', 8) % Puntos del hiperplano
plot(t, f, 'r-', 'LineWidth', 2) % Hiperplano

% Informacion de la grafica
title("El Adaline como aproximador", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)
legend('Entrada y salida deseada', 'Prediccion', 'Puntos del hiperplano', 'Hiperplano')

figure(2)
hold on
grid on

plot(J, 'g-', 'LineWidth', 2) % Grafica del error del Adaline

% Informacion de la grafica
title("Grafica del error", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
legend('error')

