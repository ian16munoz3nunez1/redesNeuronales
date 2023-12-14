% Ian Mu;oz Nu;ez - Funcion 'Logistica'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

a = 1; % Parametro que determina la pendiente de la funcion sigmoidal
phi = 1./(1 + exp(-a*v)); % Funcion o variable dependiente

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion logistica", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

