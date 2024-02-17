% Ian Mu;oz Nu;ez - MLP (Perceptron Multicapa)

close all
clear
clc

n = 30;
xl = 0; xu = 2*pi;
[X, Y] = meshgrid(linspace(xl, xu, n), linspace(xl, xu, n));
Z = 2*cos(X) - sin(Y);
x = [reshape(X, 1, []); reshape(Y, 1, [])];
y = reshape(Z, 1, []);

[model, loss] = mlp(x, y, [10, 20], 1e-1, 5000); % Objeto de tipo Multi-Layer Perceptron
yp = mlpPred(model, x); % Salida obtenida por la red
yp = reshape(yp, size(X));

figure(1)
hold on
grid on
axis 'equal'
view(-45, 30)

surface(X, Y, Z)
plot3(X, Y, yp, 'r*', 'LineWidth', 2, 'MarkerSize', 4)

figure(2)
hold on
grid on

plot(loss, 'g-', 'LineWidth', 2)
xlabel('x', 'FontSize', 15)
ylabel('y', 'FontSize', 15)
zlabel('z', 'FontSize', 15)

