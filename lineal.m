% Ian Mu;oz Nu;ez - Funcion 'Lineal'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

A = 1; % Parametro que determina la pendiente de la funcion
phi = A*v; % Funcion o variable dependiente

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion lineal", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

