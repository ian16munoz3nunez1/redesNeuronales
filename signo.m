% Ian Mu;oz Nu;ez - Funcion 'Signo'

close all
clear
clc

v = linspace(-10, 10, 1000); % Variable independiente

phi = []; % Funcion o variable dependiente
for i= 1:1000
    if v(i) > 0
        phi(i) = 1;
    elseif v(i) == 0
        phi(i) = 0;
    else
        phi(i) = -1;
    end
end

figure(1)
hold on
grid on

plot(v, phi, 'g-', 'LineWidth', 2)

title("Funcion signo", 'FontSize', 20)
xlabel('v', 'FontSize', 15)
ylabel('\phi(v)', 'FontSize', 15)

