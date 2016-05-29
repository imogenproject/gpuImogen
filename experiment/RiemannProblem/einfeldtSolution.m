function [rho v P] = einfeldtSolution(x, rho0, v0, P0, gamma, t)
% [rho v P] = einfeldtSolution(nx, rho0, v0, P0, gamma, t) generates the analytic solution of the Einfeldt
% problem consisting of Uleft = [rho0, -v0, P0] for (x < 0, t=0) and
%                      Uright = [rho0, +v0, P0] for (x > 0, t=0)
% . It returns the primitive variables evaluated at coordinates x at time t.

% Density at t > 0:
% ----\           /----
%      \         /
%       ----|----
%  A |  B   x  C  |  D
% Regions:
% A -> unperturbed left state
% B -> leftgoing rarefaction wave
% x -> contact separating fluid labelled "left" from fluid labelled "right"
% E -> rightgoing rarefaction wave
% F -> unperturbed right state

polyK = P0 / rho0^gamma;
% Find soundspeed
c = sqrt(gamma*P0/rho0);

% The normalized speed at which the slabs try to flee each other
mach = v0 / c;

% Determines which branch of solutions to use
supercritical = (mach > 2/(gamma-1));

if ~supercritical;
    % The retreat speed is low enough that adiabatic decompression can push the fluid from v0 down to v=0
    % And there is a stationary region centered on x=0
    
    m = (gamma-1)/(gamma+1);
    
    % Location of rightgoing fan's head
    xRightHead = (c+v0)*t;
    
    % Location where v0 -> 0
    xRightTail = (c+(1-gamma)*v0/2)*t;
    
    % Generate logicals that partition the solution space
    set5 = (x >= xRightHead);
    set4 = (x >= xRightTail) & (x < xRightHead);
    set3 = (x >= -xRightTail) & (x < xRightTail);
    set2 = (x >= -xRightHead) & (x < -xRightTail);
    set1 = (x < -xRightHead);
    
    rho = zeros(size(x));
    v = zeros(size(x));
    P = zeros(size(x));
    
    % Beyond right tail, initial state holds
    rho(set5) = rho0;
    v(set5)   = v0;
    P(set5)   = P0;
    
    % Right rarefaction fan solution
    xtilde = x(set4) - v0*t;
    utilde = -(1-m)*(-xtilde/t + c);
    b = (xtilde / t - utilde).^2 / gamma;
    
    rho(set4) = b.^(1/(gamma-1));
    v(set4) = utilde + v0;
    
    % Center
    bctr = (c+v0*(1-gamma)/2)^2/gamma;
    rho(set3) = bctr^(1/(gamma-1));
    v(set3) = 0;
    
    
    % Left rarefaction fan
    xtilde = x(set2) + v0*t;
    utilde = (1-m)*(xtilde/t + c);
    b = (utilde - xtilde / t).^2 / gamma;
    
    rho(set2) = b.^(1/(gamma-1));
    v(set2) = utilde - v0;
    
    rho(set1) = rho0;
    v(set1)   = -v0;
    P(set1)   = P0;
    
    P = polyK*rho.^gamma;
    
else
    % Supercritical solution:
    % One rarefaction fan spans the whole inside region

    % Boundary of nontrival region
    xHead = (mach+1)*c*t;
    tHead = t;
    
    inside = (abs(x) < xHead);

    % Normalize coordinates    
    xHat = x / (c*t*(mach+1));
    
    % Apply solution derived via self-similarity transform
    rho = rho0 * abs(xHat).^mach;
    rho(~inside) = rho0;
    
    % Velocity is trivial
    v = mach*xHat;
    v(x > xHead) = mach;
    v(x < -xHead) = -mach;
    v=v*c;
    
    % Solution remains adiabatic throughout
    P = polyK * rho.^gamma;
    
end

end
