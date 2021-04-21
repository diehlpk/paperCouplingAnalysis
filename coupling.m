close all
clear
clc

% Calculation of max values in coupling methods
a=3/4;
b=2;
l=3;
delta=1/8;

% =========================
% Mixed boundary conditions
% =========================
lambda=24;

% MDCM and MSCM
vmax = (b*b-a*a);
vmax = vmax*lambda*delta*delta/48; 
fprintf('\nMixed BC (MDCM and MSCM)\n')
fprintf('vmax = %14.12f\n',vmax)

% VHCM
vmax = (b*b-a*a)- 4*(a+b)*delta/3;
vmax = vmax*lambda*delta*delta/48; 
fprintf('\nMixed BC (VHCM)\n')
fprintf('vmax = %14.12f\n',vmax)

% =============================
% Dirichlet boundary conditions
% =============================
lambda=128/27;

% MDCM and MSCM
xmax = (a*a-b*b+2*b*l)/(2*l);
vmax = (b*b-a*a)*(l-xmax)/l - (b-xmax)*(b-xmax);
vmax = vmax*lambda*delta*delta/48; 
fprintf('\nDirichlet BC (MDCM and MSCM)\n')
fprintf('xmax = %14.12f\n',xmax)
fprintf('vmax = %14.12f\n',vmax)

% VHCM
xmax = (3*a*a-3*b*b+6*b*l+4*(a+b-l)*delta)/(6*l);
vmax = (b*b-a*a)*(l-xmax)/l - (b-xmax)*(b-xmax);
vmax = vmax - 4*(l-a-b)*delta*xmax/9 - delta*(4*a/3+delta/2);
vmax = vmax*lambda*delta*delta/48;
fprintf('\nDirichlet BC (VHCM)\n')
fprintf('xmax = %14.12f\n',xmax)
fprintf('vmax = %14.12f\n',vmax)


