function rho = CharacteristicAnalysis1D_entropy(x0, periodicity, rho0, c0, M0, gamma, e, T)
% function dif = CharacteristicAnalysis1D(x0, periodicity, rho0, c0, M0, gamma, e, T)
% Computes the propagation of disturbance e initially sampled at x0 against the given background
% Require: rho0 > 0, c0 > 0, gamma > 1, size(e) = size(x0)
% Assumes circular BCs with period x0 (remaps x(t) -> mod(x(t), periodicity)

e = e / rho0; % normalize

%origShape = size(rho0);

if numel(x0) == 1
	x0 = x0 * (1:size(e));
	disp('Scalar x0 input: Using x0*(1:numel(rho0)) as initial positions');
end

if numel(x0) ~= numel(e)
	error('Position and wave amplitude arrays must be of equal size/shape');
end

x0 = x0(:);
e  = e(:);

% Compute the exact propagation of the entropy wave packets: straight lines everywhere
x1 = x0 + M0*c0*T;

rho1 = rho0*(1 + e);

xmap = mod(x1, periodicity);

% Any appearance of disordered coordinates is just the periodic remap
% and easily removed.
[xmap, ind] = sort(xmap);
rho1 = rho1(ind);

rho = interp1(xmap, rho1, x0,'pchip','extrap');

end
