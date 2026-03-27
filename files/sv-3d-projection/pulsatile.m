clc; clear; close all;

t = linspace(0,200/120,166);
beta = 2.65; % parameter that determines the pulsatility
Nr = 3; % number of rollers
Q_bar = 3.62/60; % average flow rate in mL/s
f = 120*1.8/360; % frequency of pulse revolutions/s
p = 2;   % shape parameter

s = (1 + sin(2*pi*Nr*f*t))/2;
sp = s.^p;

A = beta*Q_bar;   % pulsation amplitude scale
Q = Q_bar + A*(sp - mean(sp));

figure;
plot(t,Q)
xlabel("Time t (s)")
ylabel("Flowrate Q (mL/s)")

% Write to .flow file
filename = 'pulsatile.flow';
fid = fopen(filename, 'w');

if fid == -1
    error('Could not open file for writing.');
end

for i = 1:length(t)
    fprintf(fid, '%.6f %.12e\n', t(i), Q(i));
end

fclose(fid);

disp(['Flow file written to: ', filename]);

% Write to .flow file
filename = 'pressure.flow';
fid = fopen(filename, 'w');

if fid == -1
    error('Could not open file for writing.');
end

for i = 1:length(t)
    fprintf(fid, '%.2f %.12e\n', t(i), 0.0);
end

fclose(fid);

disp(['Flow file written to: ', filename]);