% coagulation model 1996 RAS

% ode45 solver
% initial condition IC
IXa = 1e-6;
Xa = 1e-6;
IIa = 1e-6; % <- thrombin
II = 1000; % nM <- prothrombin
VIIIa = 1e-6;
Va = 1e-6;
APC = 1e-6;
Ia = 1e-6; % fibrin

% putting all initial condition into one variable
IC = [IXa Xa IIa II VIIIa Va APC Ia];


% tspan = [0 100]; % 0 to 100 mins - time scale
% tspan = 0:0.05:100;
options = odeset('NonNegative',1:8);
% options = odeset( ...
%     'NonNegative', 1:8, ...
%     'RelTol', 1e-7,'AbsTol', 1e-7 ...
% );
tspan = linspace(0, 100, 2000); % 0 to 100 mins - time scale
[t, y] = ode45(@ODEsystem,tspan,IC,options);
IXa = y(:,1);
Xa = y(:,2);
IIa = y(:,3);
II = y(:,4);
VIIIa = y(:,5);
Va = y(:,6);
APC = y(:,7);
Ia = y(:,8);

% save data_2k_points t y IC

figure
% hold on
plot(t,IXa);
figure
plot(t,Xa);
figure
plot(t,IIa);
figure
plot(t,II);
figure
plot(t,VIIIa);
figure
plot(t,Va);
figure
plot(t,APC);
figure
plot(t,Ia);
% legend('IXa', 'Xa', 'IIa', 'II', 'VIIIa', 'Va', 'APC', 'Ia')
% close all
% plot(t, II)

function dydt = ODEsystem(t, y)
% K[0-21] in the Biopinn
k9 = 20; % min.^-1
h9 = 0.2; % min.^-1
XIa = 0.3; % nM

k10 = 0.003; % min.^-1
k10_ = 500; % min.^-1
h10 = 1; % min.^-1

k2 = 2.3; % min.^-1
k2_ = 2000; % min.^-1
k2m = 58; % nM
k2m_ = 210; % nM
h2 = 1.3; % min.^-1

k8 = 0.00001; % min.^-1
h8 = 0.31; % min.^-1
ka = 1.2; % nM.^-1 min.^-1

k5 = 0.17; % min.^-1
h5 = 0.31; % min.^-1

k_apc = 0.0014; % min.^-1
h_apc = 0.1; % min.^-1

k1 = 2.82; % min.^-1

h11 = 0.2; % min.^-1

k5_10 = 100; % min.^-1nM^-1
k8_9 = 100; % min.^-1nM^-1
h5_10 = 100; % min.^-1
h8_9 = 100; % min.^-1

IXa = y(1); 
Xa  = y(2);
IIa = y(3);
II  = y(4);
VIIIa = y(5);
Va = y(6);
APC = y(7);
Ia = y(8);
Z = k8_9*VIIIa*IXa/(h8_9+ka*APC);
W = k5_10*Va*Xa/(h5_10+ka*APC);

% 8 ODEs 
dydt = zeros(size(y));
% dydt(1) = k9*XIa - h9*IXa;
% dydt(2) = k10*IXa + k10_*Z - h10*Xa;
% dydt(3) = k2*Xa*(II/(II+k2m)) + k2_*W*(II/(II+k2m_)) - h2*IIa;
% dydt(4) = -k2*Xa*(II/(II+k2m)) - k2_*W*(II/(II+k2m_));
% dydt(5) = k8*IIa - h8*VIIIa - ka*APC*(VIIIa+Z);
% dydt(6) = k5*IIa - h5*Va - ka*APC*(Va+W);
% dydt(7) = k_apc*IIa - h_apc*APC;
% dydt(8) = k1*IIa;

dydt(1) = k9*XIa - h9*y(1);
dydt(2) = k10*y(1) + k10_*Z - h10*y(2);
dydt(3) = k2*y(2)*(y(4)/(y(4)+k2m)) + k2_*W*(y(4)/(y(4)+k2m_)) - h2*y(3);
dydt(4) = -k2*y(2)*(y(4)/(y(4)+k2m)) - k2_*W*(y(4)/(y(4)+k2m_));
dydt(5) = k8*IIa - h8*VIIIa - ka*APC*(VIIIa+Z);
dydt(6) = k5*IIa - h5*Va - ka*APC*(Va+W);
dydt(7) = k_apc*IIa - h_apc*APC;
dydt(8) = k1*IIa;

% disp(t)

end