function flow = analyticalOutflow(r0, v0, rho0, k, r)

c0 = rho0*v0*r0;
c1 = r0*(rho0*v0*v0 + k*rho0^(5/3));

pc = [c0^3, -3*c0^2*c1, 3*c0*c1^2,-c1^3,0,0,0,0,0];

flow.r    = r;
flow.v    = zeros(size(r));
flow.rho  = zeros(size(r));
flow.ener = zeros(size(r));

if v0 == 0
    flow.rho = rho0*ones(size(r));
    flow.ener = 1.5*k*rho0^(5/3)*ones(size(r));
    return;
end

for x = 1:numel(r)
    pc(9) = k^3*c0^5/r(x)^2;
    soln = roots(pc);

    soln = soln(imag(soln) == 0);
%    if(r(x) > r0)
        v = max(soln);
%    else % I think this is incorrect in all cases...
%        v = min(soln);
%    end

    flow.v(x)    = v;
    flow.rho(x)  = c0 / (r(x)*v);
end

flow.ener = .5*flow.rho.*flow.v.^2 + 1.5*k*flow.rho.^(5/3);
flow.mom = flow.rho .* flow.v;
flow.press = 1.5*k*flow.rho.^(5/3);

end
