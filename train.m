% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

close all
clear
clc

x = [0 0 1 1;
    0 1 0 1]; % Datos de entrada
y = [0 1 1 0]; % Salida deseada

[model, loss] = mlp(x, y, [3], 0.8, 5000); % Objeto de tipo Multi-Layer Perceptron
yp = mlpPred(model, x); % Salida obtenida por la red

xl = min(min(x))-0.2; xu = max(max(x))+0.2; n = 50;
xLim = linspace(xl, xu, n);
yLim = linspace(xl, xu, n);
[X, Y] = meshgrid(xLim, yLim);

Z = mlpPred(model, [reshape(X,1,[]); reshape(Y,1,[])]);
Z = reshape(Z, size(X));

figure(1)

subplot(1,3,1)
hold on
grid on
contourf(X, Y, Z, 20)
plot(x(1,:), x(2,:), 'r*', 'MarkerSize', 8, 'LineWidth', 4)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

subplot(1,3,2)
axis 'equal'
axis([xl xu xl xu xl xu])
hold on
grid on
surf(X, Y, Z)
plot3(x(1,:), x(2,:), yp, 'r*', 'MarkerSize', 8, 'LineWidth', 4)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

subplot(1,3,3)
hold on
grid on
contour(X, Y, Z, 20, 'LineWidth', 2)
plot(x(1,:), x(2,:), 'r*', 'MarkerSize', 8, 'LineWidth', 4)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

figure(2)
hold on
grid on

plot(loss, 'g-', 'LineWidth', 2)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

