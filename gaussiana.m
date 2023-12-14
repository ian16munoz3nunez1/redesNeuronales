% Ian Mu;oz Nu;ez - Funcion 'Gaussiana'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

A = 1; % Parametro que determina la magnitud de la funcion
B = 0.1; % Parametro que determina el ancho de la funcion
phi = A*exp(-B*v.^2); % Funcion o variable dependiente

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion Gaussiana", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

