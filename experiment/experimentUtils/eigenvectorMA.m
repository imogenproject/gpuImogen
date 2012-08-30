function ev = eigenvectorMA(flow, kx, ky, kz, w)
% This function returns the fast MA eigenvector coefficients [dflow.vx dflow.vy dvz dbz dflow.by dbz]
%in terms of dflow.rho given 0th order flow and wave parameters.
% This function requires that v || b and that the flow is rotated to lie in the xy plane.

csq = flow.gamma*flow.P/flow.rho;

lambda = kx*flow.vx + ky*flow.vy - w;
kdb = kx*flow.bx + ky*flow.by;
ksq = kx^2 + ky^2 + kz^2;
bsq = flow.bx^2 + flow.by^2;
rlsq = flow.rho*lambda^2;

ev = zeros([7 1]);
%ev = [0 0 0 0 0 0 0];

% Coefficients of perturbed V[xyz]

ev(1) = 1;

ev(2) = (csq/(flow.rho*lambda))*(kx*rlsq - ksq*flow.bx*kdb)/(ksq*bsq - rlsq);
ev(3) = (csq/(flow.rho*lambda))*(ky*rlsq - ksq*flow.by*kdb)/(ksq*bsq - rlsq);
ev(4) = csq*kz*lambda/(ksq*bsq - rlsq);

% Coefficient of perturbed B[xyz]
ev(5) = csq*( kx*ky*flow.by - (ky^2+kz^2)*flow.bx)/(ksq*bsq - rlsq);
ev(6) = -csq*(-kx*ky*flow.bx + (kx^2+kz^2)*flow.by)/(ksq*bsq - rlsq);
ev(7) = csq*kz*kdb/(ksq*bsq - rlsq);
end
