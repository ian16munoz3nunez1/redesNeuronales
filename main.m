% Ian Mu;oz Nu;ez - Adaline
% Desarrollar un código en el que una neurona sea capaz de
% clasificar patrones de entrada con una función AND
% utilizando la función logística:
%           1
% y = -------------
%     1 + exp(-a*v)

close all
clear
clc

x = [0 0 1 1;
    0 1 0 1]; % Datos de entrenamiento
d = [0 0 0 1]; % Salida deseada
eta = 1e-1; % Factor de aprendizaje
epocas = 1000; % Numero de iteraciones deseadas

w = rand(size(x,1), size(d,1)); % Pesos sinapticos
b = rand(1, size(d,1)); % Bias
a = 1; % Valor para definir la pendiente de la funcion logistica

p = size(x,2); % Numero de patrones de entrada
J = zeros(1,epocas); % Arreglo para almacenar el error del Adaline

function y = logistic(a,v)
    y = 1./(1 + exp(-a*v)); % Funcion logistica aplicada el valor 'v'
end

for epoca= 1:epocas
    v = w'*x + b; % Interaccion de la entrada con los pesos y el bias
    y = logistic(a,v); % Valor que regresa la funcion logistica

    e = d - y; % Error entre la salida deseada y la obtenida
    w = w + (eta/p)*(x*e'); % Ajuste de los pesos sinapticos
    b = b + (eta/p)*sum(e); % Ajuste del valor del bias

    J(epoca) = sum((d - logistic(a,v)).^2); % Error cuadratico medio
end

y = 1*(y>0.5); % Clasificacion de la salida del Adaline

xl = min(min(x))-0.1; % Limite inferior
xu = max(max(x))+0.1; % Limite superior
v = xl:0.1:xu; % Arreglo de valores para el hiperplano separador

m = -w(1)/w(2); % Pendiente del hiperplano
b = -b/w(2); % Coeficiente del hiperplano
f = m*v + b; % Funcion del hiperplano separador

figure(1)
hold on
grid on
axis 'equal' % Muestra la escala de los ejes igual
axis([xl xu xl xu]) % Limite de los ejes 'x' y 'y'

% Grafica de los datos de entrada y su clasificacion
plot(x(1,y==1), x(2,y==1), 'y*','LineWidth', 6, 'MarkerSize', 12)
plot(x(1,y==0), x(2,y==0), 'g*', 'LineWidth', 6, 'MarkerSize', 12)
plot(v, f, 'r-', 'LineWidth', 2) % Grafica del hiperplano separador

% Informacion de la grafica
title("Neurona Logistica", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)
legend('c_0', 'c_1', 'Hiperplano')

figure(2)
hold on
grid on

plot(J, 'g-', 'LineWidth', 2) % Grafica del error

% Informacion de la grafica
title("Grafica del error", 'FontSize', 20)
xlabel('Epoca', 'FontSize', 15)
ylabel('Error', 'FontSize', 15)
legend('error')

