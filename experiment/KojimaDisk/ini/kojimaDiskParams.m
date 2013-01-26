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

% Here we go off the deep end making approximations

% Various moderately accurate approximations to R of max height
r150 = 2*(hsq-1)/(3*c1);
r200 = -(2+sqrt(4+6*c1*hsq))/(6*c1);

% Interpolate linearly; assume q >= 1.5.
% For high-q disks Zmax sits basically at rhomax
if q > 2; r = 1; end
if q <= 2.00; r = r150+2*(q-1.50)*(r200-r150); end
%if q <= 1.75; r = r150+4*(q-1.50)*(r175-r150); end
if (r < params.rin) || (r > params.rout); r = 1; end

params.height = sqrt(2*r^2 + 2*c1*r^3 - hsq*r^(5-2*q)/(q-1));
params.aspRatio = params.height / params.rout;

% The following errors were measured in this approximation of aspect ratio:
% 2.0     -14% -7%  -1.2% +1.2%
% 1.8     -18% -14% -18%  -2.4%
% 1.6     -40% -45% -7.6% -9%
% Q  \Rin .6   .7   .8    .9
% The following linear regression was found for error (pred/true - 1):
% -1.61 + .74*rin + .5*q
% Which leads to 
% Ar_corrected = params.aspRatio / (-.61+.74*rin+.5*q) to correct it.
params.aspRatio = params.aspRatio / (-.61+.74*params.rin + .5*q);
params.height   = params.aspRatio * params.rout;
params.WARNING  = 'Aspect ratio/height predictions accurate to +- ~15% error.';

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
