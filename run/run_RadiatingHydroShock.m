%   Run 3D Corrugation instability shock test.

%--- Initialize test ---%
grid = [1280 1 1];
run         = RadiatingShockInitializer(grid);

run.iterMax     = 20000;
run.theta       = 0;
run.sonicMach   = 7;
run.cfl = 0.7;
% This sets the radiation prefactor in the parameterized equation
%        \Gamma = -beta rho^2 T^theta
% It's irrelevant outside of changing output units because this parameter has a fixed relation 
% to the cooling length and the simulation automatically renormalizes dx based on the 
% fractionPreshock and fractionCold parameters.
run.radBeta = 1;

% Sets the temperature dependence of the cooling equation
% theta = 0.5 matches the classical fre-free Bremsstrahlung 
run.radTheta = .0;

% With the whole X length of the grid taken as 1, these set the length of the equilibrium
% preshock & cold gas layers; Default values are .25 and .1 respectively
% Low-theta (~ <.2) shocks undergo large-amplitude breathing modes and require a large cold gas
% layer to let the large-amplitude downstream waves propagate away
run.fractionPreshock = 0.25;
run.fractionCold     = 0.5;

% This sets the temperature relative to the preshock temperature at which radiation rate is clamped
% to zero. Physically, the value makes little sense if < 1 (since the equilibrium generator assumes
% nonradiating preshock fluid). Practically, it must be slightly more than 1, or the radiation
% cutoff at the radiating-cold point becomes numerically unhappy. Default 1.05
%run.Tcutoff = 1.05

run.bcMode.x = ENUM.BCMODE_CONST;

run.ppSave.dim2 = 5;
run.ppSave.dim3 = 100;
run.seedAmplitude = 0e-2;

run.image.interval = 20;
run.image.mass = true;
%run.image.magY = true;
%run.image.pGas = true;

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 100;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 2; % hllc
run.peripherals{end+1}=fm;

rez = run.geomgr.globalDomainRez;
run.alias       = sprintf('RHD_ms%i_ang%i', run.sonicMach, run.theta);
run.info        = sprintf('Radiating hydrodynamic shock test [Th=%g, Ms=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, rez(1), rez(2), rez(3));
%run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

