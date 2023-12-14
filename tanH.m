% Ian Mu;oz Nu;ez - Funcion 'Tangente Hiperbolica'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

phi = tanh(v); % Funcion o variable dependiente

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion tangente hiperbolica", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('$\phi(v)$', 'FontSize', 15)

