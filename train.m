% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

close all
clear
clc

clases = 6; % Numero de clases deseadas
p = 40; % Numero de patrones de entrada
x = zeros(2, p*clases); % Patrones de entrada
y = zeros(clases, p*clases); % Salida deseada

% Llenado de los patrones de entrada y la salida deseada
n = 30;
xl = -2; xu = 2;
for i= 1:clases
    seed = xl + (xu-xl)*rand(2,1);
    x(:, (p*i)-(p-1):p*i) = seed + 0.2*rand(2,p);
    y(i, (p*i)-(p-1):p*i) = ones(1,p);
end

[model, loss] = mlp(x, y, [10, 20], 1e-2, 5000); % Objeto de tipo Multi-Layer Perceptron
yp = mlpPred(model, x); % Salida obtenida por la red

colors = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1]; % Colores para diferenciar los grupos

figure(1)

subplot(131)
hold on
grid on

[_, yc] = max(y);
for i= 1:size(x,2)
    plot(x(1,i), x(2,i), 'o', 'Color', colors(yc(i),:), 'LineWidth', 8, 'MarkerSize', 8)
end
title("Problema original", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

subplot(132)
hold on
grid on

[_, yc] = max(yp);
for i= 1:size(x,2)
    plot(x(1,i), x(2,i), 'o', 'Color', colors(yc(i),:), 'LineWidth', 8, 'MarkerSize', 8)
end
title("Prediccion de la red", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

subplot(133)
hold on
grid on

n = 30;
[X, Y] = meshgrid(linspace(xl, xu, n), linspace(xl, xu, n));
x = [reshape(X, 1, []); reshape(Y, 1, [])];
yp = mlpPred(model, x);
[_, yc] = max(yp);
for i= 1:size(x,2)
    plot(x(1,i), x(2,i), 'o', 'Color', colors(yc(i),:), 'LineWidth', 8, 'MarkerSize', 8)
end
title("Areas de clasificacion", 'FontSize', 20)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)

figure(2)
hold on
grid on

plot(loss, 'g-', 'LineWidth', 2)
title("Grafica del error", 'FontSize', 20)
xlabel('Epocas', 'FontSize', 15)
ylabel('Error', 'FontSize', 15)

