% Ian Mu;oz Nu;ez - Adaline
% Desarrollar un código en el que una neurona sea capaz de
% aproximar patrones de entrada con la función logística:
%           1
% y = -------------
%     1 + exp(-a*v)

close all
clear
clc

xl = 0; xm = 10; xu = 20; n = 40;
x = [xl+(xm-xl)*rand(1,n) xm+(xu-xm)*rand(1,n)]; % Datos de entrenamiento
d = [zeros(1,n) ones(1,n)]; % Salida deseada
eta = 1e-1; % Factor de aprendizaje
epocas = 1000; % Numero de iteraciones deseadas

w = (-5)+((0)-(-5))*rand(size(x,1), size(d,1)); % Pesos sinapticos
b = (0)+((5)-(0))*rand(1, size(d,1)); % Bias
a = 1; % Valor para definir la pendiente de la funcion logistica

p = size(x,2); % Numero de patrones de entrada
J = zeros(1,epocas); % Arreglo para almacenar el error del Adaline

function y = logistic(a,v)
    y = 1./(1 + exp(-a*v)); % Funcion logistica aplicada el valor 'v'
end

t = xl:0.01:xu; % Arreglo de valores para el hiperplano

for epoca= 1:epocas
    %% Animacion
    v = w'*t + b;
    y = logistic(a,v);
    cla
    hold on
    grid on
    title(epoca, 'FontSize', 20)
    xlabel('x', 'FontSize', 15)
    ylabel('y', 'FontSize', 15)
    plot(x, d, 'b*', 'LineWidth', 4, 'MarkerSize', 4)
    plot(t, y, 'r--', 'LineWidth', 2)
    % pause(0.1)
    drawnow

    v = w'*x + b; % Interaccion de la entrada con los pesos y el bias
    y = logistic(a,v); % Salida de la funcion logistica

    e = d - y; % Error entre la salida deseada y la obtenida
    w = w + (eta/p)*(x*e'); % Ajuste de los pesos sinapticos
    b = b + (eta/p)*sum(e); % Ajuste del valor del bias

    J(epoca) = sum((d - logistic(a,v)).^2); % Error cuadratico medio

end

v = w'*t + b;
y = logistic(a,v);

figure(1)
hold on
grid on

% Grafica de los datos de entrada y su clasificacion
plot(x, d, 'b*', 'LineWidth', 4, 'MarkerSize', 4)
plot(t, y, 'r-', 'LineWidth', 2)

% Informacion de la grafica
title("Neurona Logistica", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)
legend('Datos de entrada', 'Hiperplano')

figure(2)
hold on
grid on

plot(J, 'g-', 'LineWidth', 2) % Grafica del error

% Informacion de la grafica
title("Grafica del error", 'FontSize', 20)
xlabel('Epoca', 'FontSize', 15)
ylabel('Error', 'FontSize', 15)
legend('error')

