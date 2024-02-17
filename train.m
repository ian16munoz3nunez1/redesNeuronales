% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

close all
clear
clc

n = 200;
x = linspace(0, 2*pi, n);
y = sin(x);

[model, loss] = mlp(x, y, [10, 3, 5], 1e-2, 50000); % Objeto de tipo Multi-Layer Perceptron
yp = mlpPred(model, x); % Salida obtenida por la red

figure(1)
hold on
grid on

plot(x, y, 'b-', 'LineWidth', 2)
plot(x, yp, 'r--', 'LineWidth', 2)

figure(2)
hold on
grid on

plot(loss, 'g-', 'LineWidth', 2)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

