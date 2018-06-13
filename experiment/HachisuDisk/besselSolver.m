function phi = besselSolver(rho, r, z)

z = z - mean(z); % Minimize the average |z| to maximize convergence rate

[rgrid, zgrid] = ndgrid(r,z);

n = 0;

kset = .03:.5:.5*pi/max(diff(r));
phi = zeros(size(rho));

for k = kset
    [u, zplus]  = ndgrid(r, exp(k*z));
    [u, zminus] = ndgrid(r.*besselj(0, k*r), exp(-k*z)); % get r J_n(kr) varying radially and the exponentials vertically

    va = sum(u.*rho, 1);
    vb = va .* exp(-k*z);
    va = va .* exp(k*z);

    [u, zplus]  = ndgrid(r, exp(k*z));
    [u, zminus] = ndgrid(besselj(0, k*r), exp(-k*z));

    vasum = cumsum(va);
    vbsum = sum(vb) - cumsum(vb);

    [alpha, beta] = ndgrid(r, vasum);
    [alpha, gamma]= ndgrid(r, vbsum);

    phi = phi + u .* (zminus .* beta + zplus .* gamma);

%    phi = phi + besselj(0, k*rgrid) .* exp(-k*abs(zgrid - z(20))) * besselj(0, k*r(15));

end

phi = phi * -.5;

end
