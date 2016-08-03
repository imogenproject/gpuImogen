function rho = CharacteristicAnalysis1D(x0, periodicity, rho0, c0, M0, gamma, e, T)
% function dif = CharacteristicAnalysis1D(x0, periodicity, rho0, c0, M0, gamma, e, T)
% Computes the propagation of disturbance e initially sampled at x0 against the given background
% Require: rho0 > 0, c0 > 0, gamma > 1, size(e) = size(x0)
% Assumes circular BCs with period x0 (remaps x(t) -> mod(x(t), periodicity)

e = e / rho0; % normalize

origShape = size(rho0);

if numel(x0) == 1;
	x0 = x0 * (1:size(e));
	disp('Scalar x0 input: Using x0*(1:numel(rho0)) as initial positions');
end

if numel(x0) ~= numel(e)
	error('Position and wave amplitude arrays must be of equal size/shape');
end

x0 = x0(:);
e  = e(:);

gp1 = gamma+1;
gm1 = gamma-1;

% Compute the exact propagation of the characteristic packets, assuming no
% overturn
x1 = x0 + c0*T.*((1+M0) + gp1*( (1+e).^(gm1/2) - 1 )/gm1);

% fwd diffs
de_FD = circshift(e,-1) - e;
de_BD = circshift(de_FD,1);
dx_FD = circshift(x0,-1) - x0; dx_FD = mod(dx_FD, periodicity);
dx_BD = circshift(dx_FD,1);
% Extract b from the (a + bx + cx^2) van Der Monde fit
% to know e' to second order accuracy
de_dx = de_FD.*dx_FD.*dx_FD;
de_dx = de_dx + circshift(de_dx,1);
de_dx = de_dx ./ ( dx_FD.*dx_BD.*(dx_FD+dx_BD));

% Compute the critical time
criticalTime = 2/(gp1*c0*max(de_dx));

if any(T > criticalTime)
    warning('WARNING: T > critical time: SHOCK HAS FORMED, CHARACTERISTIC PROJECTION INVALID');
end

rho1 = rho0*(1 + e);

xmap = mod(x1, periodicity);

% Assuming we have not surpassed the critical time,
% any appearance of disordered coordinates is just the periodic remap
% and easily removed.
[xmap ind] = sort(xmap);
rho1 = rho1(ind);

rho = interp1(xmap, rho1, x0,'pchip','extrap');

end
