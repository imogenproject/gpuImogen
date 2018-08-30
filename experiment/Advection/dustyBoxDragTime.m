function Kdrag = dustyBoxDragTime(fluidDetails, rhoG, rhoD, vGas, vDust, gammaGas, Pgas)
% Inefficiently but very readably computes the relative accelerations of the
% gas and dust fluids for a fully general drag law.

sigmaGas = fluidDetails(1).sigma;
muGas    = fluidDetails(1).mass;
sigmaDust= fluidDetails(2).sigma;
muDust   = fluidDetails(2).mass;


alpha   = 128.0 * sigmaGas * sqrt(sigmaDust) / (5*muGas*pi*sqrt(gammaGas-1));
beta    = 128.0 * (gammaGas-1)/(9*pi);
epsilon = 1/gammaGas;
theta   = 5*pi*sqrt(pi/2)*muGas / (144*sigmaGas);

dv0 = vGas - vDust;
dv = dv0; % makes an analytic extrapolation for uinternal go away 

sg = sign(dv);
dv = abs(dv);

% initial internal energy
u0 = (Pgas / (gammaGas-1)) / rhoG;

% make sure computation includes gas heating term!
% what we're "really" want is c_s, so use only translational Dof (gamma-1)
uinternal = u0 + .5*(gammaGas-1)*rhoD * (dv0^2 - dv^2) / (rhoG + rhoD);

kEpstein = sqrt(beta*uinternal + epsilon * dv^2);

% FIXME this implementation is poorly conditioned (re ~ 1/v for v << v0)
Re = alpha * dv * rhoG / sqrt(uinternal);
kStokes = stokesDragCoeff(Re) * dv;
if Re < 1e-6
    kStokes = 12 * sqrt(uinternal) / (alpha*rhoG);
end
        
sigma0 = (theta  / rhoG)^2; % sqrt(pi)*(4 l_mfp / 9) = sqrt(pi) * s0
                            % = pi s0^2 = epstein/stokes cutover crossection
Kdrag = ( sigma0 * kEpstein + sigmaDust * kStokes) * sigmaDust * (rhoG + rhoD) / (muDust * (sigma0 + sigmaDust));

end
