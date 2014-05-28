function [result] = HDJumpSolver(ms, theta, gamma)
% Solve the 3D Hydrodynamic jump conditions for an equilibrium shock wave, provided that the
% preshock region conforms to zero z component of momentum. Takes the shock as existing in the
% YZ plane with flow in the positive X direction; Your coordinate system may vary.
%
% Solution otherwise conforms to any initial conditions that support a shock.
%
% Note that solutions are returned in the shock-static frame.

theta = theta * pi / 180;

% Normalize away
rho1 = 1;
P1 = 1;

cs1 = sqrt(gamma);
gm1 = gamma - 1;
gp1 = gamma + 1;

vx1 = ms*cs1;
vy1 = vx1 * tan(theta);

if ms <= 1
    % The only solution is no jump
    result.rho = [1 1];
    result.P = [1 1];
    result.v = [vx1 vx1; vy1 vy1];
    result.error = [0 0 0];

    return;
end

% Preshock kinetic energy density
T1 = rho1 * (vx1^2 + vy1^2) / 2;

  vx2 = vx1*gm1/gp1 + 2*gamma*P1/(vx1*rho1*gp1);
  vy2 = vy1;
  rho2 = rho1*vx1/vx2;

  T2 = rho2*(vx2^2 + vy2^2)/2;
  P2 = (2*rho1*vx1^2 - gm1*P1)/(gp1);

  result.rho = [rho1 rho2];
  result.v = [vx1 vx2; vy1 vy2];
  result.B = [0 0; 0 0; 0 0]; % For compatibility w/MHDJumpSolver output
  result.Pgas = [P1 P2];
  result.Etot = [P1/(gamma-1) + T1, P2/(gamma-1) + T2];
  result.theta = theta;
  result.sonicMach = ms;
  result.error = [rho2*vx2 - rho1*vx1, (rho1*vx1^2 + P1 - (rho2*vx2^2+P2)), vx1*(T1+gamma*P1/gm1) - vx2*(T2+gamma*P2/gm1)];

end

