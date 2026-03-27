clear; clc; close all;

%% Parameters
mu = 7e-4;          % viscosity [Pa.s]
L  = 4.592e-2;         % straight branch length [m]
Q  = 6e-8;          % volumetric flow rate [m^3/s]
N  = 400;           % number of axial points
x  = linspace(0,L,N);

% Baseline radii [m]
% rt_left  = 0.5-3/2;   % top radius at left
% rt_right = 0.20e-3/2;   % top radius at right
% 
% rb_left  = 0.05e-3/2;   % bottom radius at left
% rb_right = 0.30e-3/2;   % bottom radius at right

% You can flip these values to test reverse taper.
% Example for widening to the right:
rt_left = 0.10e-3/2; rt_right = 0.50e-3/2;
rb_left = 0.05e-3/2; rb_right = 0.30e-3/2;

%% Linear taper definitions
rt = rt_left  + (rt_right - rt_left )*(x/L);   % top radius vs x
rb = rb_left  + (rb_right - rb_left )*(x/L);   % bottom radius vs x

%% Local hydraulic resistance per unit length
% For a circular tube: dR/dx = 8*mu / (pi*r^4)
dRdx_t = 8*mu ./ (pi*rt.^4);
dRdx_b = 8*mu ./ (pi*rb.^4);

%% Pressure profiles
% Choose arbitrary reference pressure at the left end of the top branch
P_top_left = 0;

% Top branch flows left -> right, so pressure decreases with x
% dP/dx = -Q * dRdx_t
P_top = P_top_left - cumtrapz(x, Q*dRdx_t);

% Bottom branch flows right -> left, so pressure increases with x
% Let bottom-left pressure equal pressure after the U-bend offset.
% For simplicity here, use the same left-end reference.
P_bot_left = 0;
P_bot = P_bot_left + cumtrapz(x, Q*dRdx_b);

%% Pressure difference between bottom and top at same x
DeltaP = P_bot - P_top;

%% Metrics of uniformity
DeltaP_range = max(DeltaP) - min(DeltaP);
DeltaP_std   = std(DeltaP);

fprintf('DeltaP range = %.4e Pa\n', DeltaP_range);
fprintf('DeltaP std   = %.4e Pa\n', DeltaP_std);

%% Plot radii
figure;
plot(x*1e3, rt*1e3, 'LineWidth', 2); hold on;
plot(x*1e3, rb*1e3, 'LineWidth', 2);
xlabel('x [mm]');
ylabel('Radius [mm]');
legend('Top radius','Bottom radius','Location','best');
title('Channel taper');
grid on;

%% Plot pressures
figure;
plot(x*1e3, P_top, 'LineWidth', 2); hold on;
plot(x*1e3, P_bot, 'LineWidth', 2);
xlabel('x [mm]');
ylabel('Pressure [Pa]');
legend('Top channel','Bottom channel','Location','best');
title('Pressure profiles');
grid on;

%% Plot DeltaP
figure;
plot(x*1e3, DeltaP, 'LineWidth', 2);
xlabel('x [mm]');
ylabel('\DeltaP = P_{bottom} - P_{top} [Pa]');
title('\DeltaP along channel length');
grid on;