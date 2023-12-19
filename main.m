% Ian Mu;oz Nu;ez - Perceptron
% Diseñar una red neuronal unicapa, con función tipo escalón, con dos
% entradas y dos salidas que sea capaz de clasificiar los siguientes
% diez puntos en el plano, los cuales pertenecen a cuatro grupos:
% Grupo 1: (0.1, 1.2), (0.7, 1.8), (0.8, 1.6)
% Grupo 2: (0.8, 0.6), (1.0, 0.8)
% Grupo 3: (0.3, 0.5), (0.0, 0.2), (-0.3, 0.8)
% Grupo 4: (-0.5, -1.5), (-1.5, -1.3)

close all
clear
clc

x = [0.1 0.7 0.8 0.8 1.0 0.3 0.0 -0.3 -0.5 -1.5;
    1.2 1.8 1.6 0.6 0.8 0.5 0.2 0.8 -1.5 -1.3]; % Datos de entrada
b = [1 1 1 1 1 1 1 1 1 1]; % Entrada fija para el bias
x = [x; b]; % Datos de entrenamiento con entrada fija
d = [0 0 0 0 0 1 1 1 1 1;
    0 0 0 1 1 0 0 0 1 1]; % Salida deseada
w = rand(3,2); % Pesos sinapticos aleatorios iniciales
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
        for j= 1:size(d, 1)
            v = w(:,j)' * x(:,i); % Multiplicacion del vector de entrada por el vector de pesos sinapticos
            y(j,i) = escalon(v); % Funcion de activacion

            e = d(j,i) - y(j,i); % Error obtenido
            if e < 0 || e > 0
                w(:,j) = w(:,j) + e*x(:,i); % Ajuste de pesos
                ep = ep + 1;
            end
        end
    end

    % Si no hubo ningun error en el entrenamiento, se termina el proceso
    if ep == 0
        break
    end
end

xl = min(min(x))-0.1; % Limite inferior para mostrar la grafica
xu = max(max(x))+0.1; % Limite superior para mostrar la grafica
t = xl:0.1:xu; % Arreglo de valores para el hiperplano separador

f1 = -(w(1,1)/w(2,1))*t - (w(3,1)/w(2,1)); % Funcion del hiperplano 1
f2 = -(w(1,2)/w(2,2))*t - (w(3,2)/w(2,2)); % Funcion del hiperplano 2

figure(1)
hold on
grid on
axis equal % Muestra la escala de los ejes igual
axis([xl xu xl xu]) % Limite de los ejes

% Grafica de los datos de entrada y su clasificacion
plot(x(1, (y(1,:)==1)&(y(2,:)==1)), x(2, (y(1,:)==1)&(y(2,:)==1)), 'r*', 'LineWidth', 2, 'MarkerSize', 8)
plot(x(1, (y(1,:)==0)&(y(2,:)==1)), x(2, (y(1,:)==0)&(y(2,:)==1)), 'b*', 'LineWidth', 2, 'MarkerSize', 8)
plot(x(1, (y(1,:)==1)&(y(2,:)==0)), x(2, (y(1,:)==1)&(y(2,:)==0)), 'g*', 'LineWidth', 2, 'MarkerSize', 8)
plot(x(1, (y(1,:)==0)&(y(2,:)==0)), x(2, (y(1,:)==0)&(y(2,:)==0)), 'y*', 'LineWidth', 2, 'MarkerSize', 8)
plot(t, f1, 'm-', 'LineWidth', 2) % Grafica del hiperplano separador 1
plot(t, f2, 'c-', 'LineWidth', 2) % Grafica del hiperplano separador 2

% Informacion de la grafica
title("A(B+C)", 'FontSize', 20)
xlabel('A', 'FontSize', 15)
ylabel('B', 'FontSize', 15)
legend('c_0', 'c_1', 'c_2', 'c_3', 'Hiperplano 1', 'Hiperplano 2')

