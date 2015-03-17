function [rho, P, v] = ShuOsherLinearSoln(gamma, Mach, ampEnt_in, xcoords, Kin, t)
% Given the fluid gamma, incoming Mach, and entropy wave amplitude with wavevector k
% Solves the Shu-Osher problem exactly to first order in ampEnt_in.
% Gives answer quantities in the S-O problem frame
% (static preshock fluid, rightgoing shock) at the given time t.

% This solver normalizes to rhopre = Ppre = 1
c1  = sqrt(gamma);
% First step: Solve the jump equations from preshock to postshock at 0th order
equil = HDJumpSolver(Mach, 0, gamma);

% Extract the parts used by the Mathematica generated solution to short names
c2   = sqrt(gamma*equil.Pgas(2)/equil.rho(2));

v1   = equil.v(1,1);
rho1 = equil.rho(1);
P1   = equil.Pgas(1);

v2   = equil.v(1,2);
rho2 = equil.rho(2);
P2   = equil.Pgas(2);

% We started with incoming kx wavevector so find the omega from preshock entropy wave dispersion reln
omega = Kin*equil.v(1,1);

% Outgoing sonic & entropy wave k vectors 
Ksonic = omega / (equil.v(1,2) + c2);
Kent   = omega / (equil.v(1,2));

% Some simplifiers...
gm1 = gamma - 1.0; gp1 = gamma + 1.0;

sndamplitude = (2*ampEnt_in*(v1^2)*(gamma*P1 - gamma*P2 + gm1*rho1*v1*(v1 - v2))*(v1 - v2))/(c2*(c2*v1*(2*gamma*P1 - 2*gamma*P2 + rho1*(v1 - v2)*(3*gm1*v1 - gp1*v2)) + v2*(2*gamma*P1*v1 + 2*gamma*P2*(-2*v1 + v2) + gm1*rho1*v1*(3*(v1^2) - 4*v1*v2 + (v2^2)))));

entamplitude = (ampEnt_in*(v1^2)*(-2*(gamma*P1 - gamma*P2 + gm1*rho1*v1*(v1 - v2))*(v1 - v2)*(v2^2) + (c2^2)*(2*gamma*P1*v2 - 2*gamma*P2*v2 + rho1*v1*(v1 - v2)*(gm1*v1 + (-3 + gamma)*v2)) + c2*v2*(-2*gamma*P1*(v1 - 2*v2) - 2*gamma*P2*v2 - gm1*rho1*v1*((v1^2) - 4*v1*v2 + 3*(v2^2)))))/(c2*(v2^2)*(c2*v1*(2*gamma*P1 - 2*gamma*P2 + rho1*(v1 - v2)*(3*gm1*v1 - gp1*v2)) + v2*(2*gamma*P1*v1 + 2*gamma*P2*(-2*v1 + v2) + gm1*rho1*v1*(3*(v1^2) - 4*v1*v2 + (v2^2)))));

xshockamp    = (1i*ampEnt_in*v1*(v1 - v2)*(-2*gamma*P2*v2 + rho1*v1*(gm1*(v1 - v2)*v2 + c2*(gm1*v1 - gp1*v2))))/(rho1*(c2*v1*(2*gamma*P1 - 2*gamma*P2 + rho1*(v1 - v2)*(3*gm1*v1 - gp1*v2)) + v2*(2*gamma*P1*v1 + 2*gamma*P2*(-2*v1 + v2) + gm1*rho1*v1*(3*(v1^2) - 4*v1*v2 + (v2^2))))*omega);

% FIXME HACK: first we do an affine / parity xform since the Shu-Osher problem is set up
% with the shock initially at x=.25 and moving right
xcoords = .125 - xcoords;

% We now have the exact linear solution
% In coordinates comoving with the shock frame.
% with transform x = z - (v1 t + xshockamp e^-i w t)
% or,            z = x + v1 t + Re[xshockamp e^-i w t]
% (xform: zdot = v1 - Re[i w xshockamp e^-i w t])

z = xcoords + v1*t + real(xshockamp*exp(-1i*omega*t));

% I'd like to have it stick in a double-counted discontinuity at z=0 in the future...
setA = (z < 0);
setB = (z>= 0);
setC = (z < v2*t); % postshock entropy wave extent
setD = (z < (c2+v2)*t); % postshock sonic wave extent

rho = zeros(size(xcoords));
P = rho; v = rho;

% We assume that the preshock entropy wave exists forever...
% IN Z (shock static) COORDINATES,
%	at z < 0:
% rho = 1 + ampEnt_in Re[e^i(Kin z - w t)]
q = 1 + real(ampEnt_in*exp(1i*Kin*z - 1i*omega*t));
	rho(setA) = q(setA);
% P   = 1
P(setA) = 1;
% v   = v1 - Re[-i w xshockamp e^(-i w t)]
v(setA) = v1 - real(-1i*omega*xshockamp*exp(-i *omega*t));

%	at z > 0:
% rho = rho2 + Re[sndamplitude e^i(Ksonic z - w t) + entamplitude e^i(Kent z - w t)]
q = rho2 + real(sndamplitude*exp(1i*Ksonic*z - 1i*omega*t).*setD + entamplitude*exp(1i*Kent*z - 1i*omega*t).*setC);
	rho(setB) = q(setB);
% P   = P2 + Re[sndamplitude c2^2 e^i(Ksonic z - w t)]
q = P2 + real(sndamplitude*c2^2 * exp(1i*Ksonic*z - 1i*omega*t).*setD);
	P(setB) = q(setB);
% v   = v2 + Re[sndamplitude c2/rho2 e^(i Ksonic z - w t) -i w xshockamp e^(-i w t)]
q = v2 + real(sndamplitude*c2*exp(1i*Ksonic*z - 1i*omega*t).*setD/rho2 - 1i*omega*xshockamp*exp(-1i*omega*t));
	v(setB) = q(setB);

v = v - (v1 - real(-1i*omega*xshockamp*exp(-1i*omega*t)));
% map back using 
% x mapped to z and velocities transformed by given formulae


