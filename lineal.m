% Ian Mu;oz Nu;ez - Funcion 'Lineal'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

A = 1; % Parametro que determina la pendiente de la funcion
y = A*v; % Funcion o variable dependiente
dy = A*ones(1,1000); % Derivada de la funcion

figure(1)

subplot(2,1,1)
hold on
grid on

plot(v, y, 'g-', 'LineWidth', 2)
title("Funcion lineal", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

subplot(2,1,2)
hold on
grid on

plot(v, dy, 'r-', 'LineWidth', 2)
title("Derivada de la funcion", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('d \phi(v)', 'FontSize', 15)

