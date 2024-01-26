% Ian Mu;oz Nu;ez - Funcion 'Gaussiana'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

A = 1; % Parametro que determina la magnitud de la funcion
B = 0.1; % Parametro que determina el ancho de la funcion
y = A*exp(-B*v.^2); % Funcion o variable dependiente
dy = -2*A*B*v.*exp(-B*v.^2); % Derivada de la funcion

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion Gaussiana", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

