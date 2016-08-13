function params = kojimaDiskParams(q, rin, gamma)

params.q      = q;
params.rin    = rin;
params.rout   = kojimaFindOuterRadius(q, params.rin);
  
xq            = 2*(1-q);
hsq           = xq * (1/params.rout - 1/params.rin)/(params.rin^xq - params.rout^xq);
c1            = -1/params.rin - (hsq/xq)*(params.rin^xq);

% Calculate maximum density, c_s at rhomax,
params.rhomax = density(c1, hsq, xq, 1, 0, gamma);
params.Pmax   = params.rhomax^gamma;
params.csmax  = sqrt(gamma*params.Pmax/params.rhomax);

params.height = kojimaDiskHeight(q, rin, gamma);
params.aspRatio = params.height / params.rout;

rhob = @(r,z) r.*density(c1, hsq, xq, r, z, gamma);

params.Mdisk  = 4*pi*real(dblquad(rhob, params.rin, params.rout, 0, params.height));

    %--- Calculates the pressure integral (Bernoulli equation), clamps #s < 0, solves for rho ---%
%bernoulli   = c1 + (1 ./ rdinf.centerRadius) + (hsq / xq) * rdinf.axialRadius.^xq;
%rho                = ((bernoulli * (GAMMA-1)/GAMMA).^(1/(GAMMA-1))

end

function h = height(c1, hsq, xq, r)
  h = sqrt((c1 - hsq*r.^xq / xq).^-2 - r.^2);
end

function rho = density(c1, hsq, xq, r, z, gamma)
  B = bernoulli(c1, hsq, xq, r, z);
  rho = (B*(gamma-1)/gamma).^(1/(gamma-1));
end

function B = bernoulli(c1, hsq, xq, r, z)
  B = c1 + 1./sqrt(r.^2+z.^2) + hsq*(r.^xq)/xq;
end
