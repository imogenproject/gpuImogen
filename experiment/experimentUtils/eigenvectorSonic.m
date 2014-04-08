function ev = eigenvectorSonic(rho, csq, v, b, k, direct)


% This function returns the fast MA eigenvector coefficients [1 dvx dvy dvz dbx dby dbz]
% associated with the plane wave posessing the given real k.
% wavetype = [-2: fast bkwd, -1: slow bkwd, 1: slow fwd, 2: fast fwd]
% where forward == re[w] > 0 and backwards == re[w] < 0
% for exp[i(k.r - wt)]

% Rotate into the plane such that vz/bz vanish

% Solve the dispersion relation

lambda = dot(k,v) - w;



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
