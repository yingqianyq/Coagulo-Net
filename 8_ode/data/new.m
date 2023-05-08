% coagulation model 1996 RAS

% ode45 solver
% initial condition IC
IXa = 1e-7;
Xa = 1e-6;
IIa = 1e-7; % <- thrombin
II = 5; % nM <- prothrombin
VIIIa = 1e-2;
Va = 1e-6;
APC = 1e-4;
Ia = 1e-7; % fibrin

% putting all initial condition into one variable
IC = [IXa Xa IIa II VIIIa Va APC Ia];

tspan = linspace(0, 10, 2000);
[t, yy] = ode45(@ODEsystem,tspan,IC);
IXa = yy(:,1);
Xa = yy(:,2);
IIa = yy(:,3);
II = yy(:,4);
VIIIa = yy(:,5);
Va = yy(:,6);
APC = yy(:,7);
Ia = yy(:,8);

save solution_2 t yy IC

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
% scaling constants
s1 = 10;
s2 = 1;
s3 = 10;
s4 = 10;
s5 = 0.0001;
s6 = 1;
s7 = 0.01;
s8 = 10;

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

k5_10 = 100; % min.^-1nM^-1
k8_9 = 100; % min.^-1nM^-1
h5_10 = 100; % min.^-1
h8_9 = 100; % min.^-1

Z = k8_9*s5*s1*y(5)*y(1)/(h8_9+ka*y(7)*s7);
W = k5_10*s6*s2*y(6)*y(2)/(h5_10+ka*y(7)*s7);

% 8 ODEs 
dydt = zeros(size(y));

dydt(1) = k9*XIa/s1 - h9*y(1);
dydt(2) = k10*y(1)*s1/s2 + k10_*Z/s2 - h10*y(2);
dydt(3) = k2*y(2)*y(4)*s2*s4/s3/(s4*y(4)+k2m) + k2_*W*s4*y(4)/s3/(s4*y(4)+k2m_) - h2*y(3);
dydt(4) = -k2*s2*y(2)*y(4)/(s4*y(4)+k2m) - k2_*W*y(4)/(y(4)*s4+k2m_);
dydt(5) = k8*y(3)*s3/s5 - h8*y(5) - ka*y(7)*s7*(y(5)+Z/s5);
dydt(6) = k5*y(3)*s3/s6 - h5*y(6) - ka*y(7)*s7*(y(6)+W/s6);
dydt(7) = k_apc*y(3)*s3/s7 - h_apc*y(7);
dydt(8) = k1*y(3)*s3/s8;

% disp(t)

end