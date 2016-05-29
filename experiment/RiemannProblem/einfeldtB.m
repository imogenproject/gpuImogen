function [rho v P] = einfeldtB(x, rho0, v0, P0, gamma, t)
% [rho v P] = einfeldtB(nx, rho0, v, P0) is my attempt to generate the analytic solution of the 
% Einfeldt problem consisting of Uleft = [rho0, -v, P0] for (x < 0, t=0) and
%                                Uright = [rho0, +v, P0] for (x > 0, t=0)
% . It returns the primitive variables evaluated at coordinates x at time t.
% This is for the case of 'supercritical mach' in which there is no stationary central
% region separating the two fans.
% THIS MODE DOES NOT WORK YET


% Density at t > 0: 
% ----\           /----
%      \         /
%       ----|----
%  A |  B   x  C  |  D
% Regions:
% A -> unperturbed left state
% B -> leftgoing rarefaction wave
% x -> psuedo-contact separating fluid labelled "left" from fluid labelled "right"
% E -> rightgoing rarefaction wave
% F -> unperturbed right state

% Within rarefaction wave, v = x/t for x in [ -(v0+c)t, (v0+c)t ]
% Comoving, div(v) = -1/t -> rho2 = rho1 (t1 / t2)

% Find soundspeed
c = sqrt(gamma*P0/rho0);

m = v0/c;
polyK = P0/rho0^gamma;

% Location of rightgoing fan's head
xHead = (c+v0)*t

% Generate logicals that partition the solution space
set3 = (x >= xHead);
set2 = (x >= -xHead) & (x < xHead);
set1 = (x < -xHead);

rho = zeros(size(x));
v = zeros(size(x));
P = zeros(size(x));

% Beyond right tail, initial state holds
rho(set3) = rho0;
v(set3)   = v0;
P(set3)   = P0;

% Central fan
xtilde = x(set2) - v0*t;
utilde = -m*(-xtilde/t + c)/(m+1);

b = (xtilde / t - utilde).^2 / gamma;

rho(set2) = b.^(1/(gamma-1));
v(set2) = utilde + v0;

% Left initial state
rho(set1) = rho0;
v(set1)   = -v0;
P(set1)   = P0;

P = polyK*rho.^gamma;

end
