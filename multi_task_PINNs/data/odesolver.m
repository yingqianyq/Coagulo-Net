%% initial condition
% y0 = [1.0; 0.8; 0.5];
y0 = randn(3, 1);

%% time domain
t = linspace(0, 10, 501);

%% ODE solver
[t, y] = ode45(@ko, t, y0);

%% figures
figure;

plot(t, y(:, 1));
hold on
plot(t, y(:, 2));
plot(t, y(:, 3));

save ko_data_2 t y


function dydt = ko(t, y)
dydt = [y(1).*y(3);
        -y(2).*y(3);
        -y(1).^2 + y(2).^2];
end