function S = DustyBoxSolver(sigmaGas, muGas, sigmaDust, muDust, rhoG, rhoD, vGas, vDust, gammaGas, Pgas, tFinal)
% S = dustyBoxSolver(sigmaGas, muGas, sigmaDust, muDust, rhoG, rhoD, vGas, vDust, gammaGas, Pgas, tFinal)
% > sigmaGas:  Gas particle geometric cross section (molecular H2 = 2.348e-19 m^2)
% > muGas:     Gas mean molecular mass              (molecular H2 = 3.89e-27 kg)
% > sigmaDust: Dust particle avg geometric x-section
% > muDust:    Dust particle avg mass
% > rhoG:      Gas mass density
% > rhoD:      Dust mass density
% > vGas:      Scalar gas velocity
% > vDust:     Scalar dust velocity
% > gammaGas:  Gas adiabatic index
% > Pgas:      Gas hydrostatic pressure
% > tFinal:    Time to end integration at
F = @(t, y) dustyBoxDrag(sigmaGas, muGas, sigmaDust, muDust, rhoG, rhoD, vGas, vDust, gammaGas, Pgas, y);

eig = F(0, vGas-vDust)/abs(vGas-vDust);

h = .5*tFinal/eig;
n0 = ceil(tFinal / h);
h = tFinal / n0; % get an int # of steps the lazy way

S = ODESolver();
S.setODE(F);
%S.setMethod([1 1 1 1], S.methodImplicit); 
S.setMethod([1 1 1 1], S.methodExplicit); 
S.setInitialCondition(0, vGas-vDust);
S.setStep(h);
%Where @f is the ode y'=f(x,y) to solve, D (if true) indicates that the function returns a vector of (not approximated) derivatives of f equal in number to that required by the method matrix, x0, y0 and h0 give initial conditions and step size, method is a HCAM/HCAB constraint matrix and methType is ODESolver.methodExplicit or ODESolver.methodImplicit
S.integrate(tFinal);

end


function accel = dustyBoxDrag(sigmaGas, muGas, sigmaDust, muDust, rhoG, rhoD, vGas, vDust, gammaGas, Pgas, dv)


alpha   = 128.0 * sigmaGas * sqrt(sigmaDust) / (5*muGas*pi*sqrt(gammaGas-1));
beta    = 128.0 * (gammaGas-1)/(9*pi);
epsilon = 1/gammaGas;
theta   = 5*pi*sqrt(pi/2)*muGas / (144*sigmaGas);

dv0 = vGas - vDust;

sg = sign(dv);
dv = abs(dv);

% initial internal energy
u0 = (Pgas / (gammaGas-1)) / rhoG;

% make sure computation includes gas heating term!
uinternal = u0 + rhoD * (dv0^2 - dv^2) / (rhoG + rhoD);

kEpstein = sqrt(beta*uinternal + epsilon * dv^2);

% FIXME this implementation is poorly conditioned (re ~ 1/v for v << v0)
Re = alpha * dv * rhoG / sqrt(uinternal);
kStokes = stokesDragCoeff(Re) * dv;
	
sigma0 = (theta  / rhoG)^2; % sqrt(pi)*(4 l_mfp / 9) = sqrt(pi) * s0
                            % = pi s0^2 = epstein/stokes cutover crossection

accel = -sg*( sigma0 * kEpstein + sigmaDust * kStokes) * dv * sigmaDust * (rhoG + rhoD) / (muDust * (sigma0 + sigmaDust));

end

function C = stokesDragCoeff(Re)

if (Re < 1)
	C = 12 / Re;
	return;
end
if (Re > 800)
	C = 0.22;
	return;
end

C = 12 * Re^-0.6;

end
