% Ian Mu;oz Nu;ez - Perceptron
% - Construya la tabla de verdad para la siguiente función
%   lógica: A(B+C)
% - Escriba un código para implementar la función
%       +-------------------+
%       | A | B | C | F | b |
%       |---+---+---+---+---|
%       | 0 | 0 | 0 | 0 | 1 | <-- Patron 1
%       |---+---+---+---+---|
%       | 0 | 0 | 1 | 0 | 1 | <-- Patron 2
%       |---+---+---+---+---|
%       | 0 | 1 | 0 | 0 | 1 | <-- Patron 3
%       |---+---+---+---+---|
%       | 0 | 1 | 1 | 0 | 1 | <-- Patron 4
%       |---+---+---+---+---|
%       | 1 | 0 | 0 | 0 | 1 | <-- Patron 5
%       |---+---+---+---+---|
%       | 1 | 0 | 1 | 1 | 1 | <-- Patron 6
%       |---+---+---+---+---|
%       | 1 | 1 | 0 | 1 | 1 | <-- Patron 7
%       |---+---+---+---+---|
%       | 1 | 1 | 1 | 1 | 1 | <-- Patron 8
%       +-------------------+

close all
clear
clc

x = [0 0 0 0 1 1 1 1;
    0 0 1 1 0 0 1 1;
    0 1 0 1 0 1 0 1]; % Datos de entrada
b = [1 1 1 1 1 1 1 1]; % Entrada fija para el bias
x = [x; b]; % Datos de entrenamiento con entrada fija
d = [0 0 0 0 0 1 1 1]; % Salida deseada
w = rand(4,1); % Pesos sinapticos aleatorios iniciales
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
xLim = linspace(xl, xu, 100); % Valores en X para generar la malla
yLim = linspace(xl, xu, 100); % Valores en Y para generar la malla
[X, Y] = meshgrid(xLim, yLim); % Malla con rangos de valores de xl a xu para el hiperplano separador

m = -((w(1)*X)/w(3)) - ((w(2)*Y)/w(3)); % Pendiente del hiperplano
b = -w(4)/w(3); % Coeficiente del hiperplano
f = m + b; % Funcion del hiperplano separador
% f = -((w(1)*X)/w(3)) - ((w(2)*Y)/w(3)) - (w(4)/w(3)); % Funcion del hiperplano separador

figure(1)
hold on
grid on
axis equal % Muestra la escala de los ejes igual
axis([xl xu xl xu xl xu]) % Limite de los ejes

% Informacion de la grafica
title("A(B+C)", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)
zlabel('C', 'FontSize', 15)

% Grafica de los datos de entrada y su clasificacion
plot3(x(1,y==1), x(2,y==1), x(3,y==1), 'y*', 'LineWidth', 6, 'MarkerSize', 12)
plot3(x(1,y==0), x(2,y==0), x(3,y==0), 'g*', 'LineWidth', 6, 'MarkerSize', 12)
surf(X, Y, f) % Grafica del hiperplano separador

