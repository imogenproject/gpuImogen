function out = ShuOsherLinearCoeffs(gamma, Mach, ampEnt_in, Kin)
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

% Store linear response coefficients
out.xshock = xshockamp;
out.inamp  = ampEnt_in;
out.entamp = entamplitude;
out.sndamp = sndamplitude;

% Store wavevectors
out.Kin = Kin;
out.Kent = Kent;
out.Ksonic = Ksonic;

% Store shock speed
out.shockEqVelocity = v1;

end
