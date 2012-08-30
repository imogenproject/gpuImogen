function ev = eigenvectorAlfven(flow, kx, ky, kz, w)
% This function returns the Alfven eigenvector coefficients [dvx vy dbz dby dbz] in terms of dvz given 0th order flow & wave parameters
% This function requires that v || b and that the flow is rotated to lie in the xy plane.

csq = flow.gamma*flow.P/flow.rho;

lambda = kx*flow.vx + ky*flow.vy - w;
kdb = kx*flow.bx + ky*flow.by;
ksq = ky^2 + kz^2;

if min(abs(lambda - kdb/sqrt(flow.rho)), abs(lambda + kdb/sqrt(flow.rho))) > 1e-6; warning('kx to Alfven eigenvector solver likely wrong.\n'); end

% The perturbed B for the Alfven wave switches sign depending on whether the wavevector is aligned or antialigned with the magnetic field.
dirSwitch = sign(real(lambda / kdb));

ev = zeros([7 1]);

ev(1) = 0;

ev(2) = flow.by*kz / (flow.bx*ky - flow.by*kx);
ev(3) = flow.bx*kz / (flow.by*kx - flow.bx*ky);
ev(4) = 1;

ev(5) = -dirSwitch*flow.by*kz*sqrt(flow.rho)/(flow.by*kx-flow.bx*ky);
ev(6) =  dirSwitch*flow.bx*kz*sqrt(flow.rho)/(flow.by*kx-flow.bx*ky);
ev(7) =  dirSwitch*sqrt(flow.rho);

end
