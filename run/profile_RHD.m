%   Run 3D Corrugation instability shock test.

%--- Initialize test ---%
grid = [1024 256 1];
run         = RadiatingShockInitializer(grid);

run.iterMax     = 5;
run.theta       = 0;
run.sonicMach   = 8;

run.machY_boost = 0;

% This sets the radiation prefactor in the parameterized equation
%        \Gamma = -beta rho^2 T^theta
% It's irrelevant outside of changing output units because this parameter has a fixed relation 
% to the cooling length and the simulation automatically re-scales dx based on the 
% fractionPreshock and fractionCold parameters.
run.radBeta = 1;

% Sets the temperature dependence of the cooling equation
% theta = 0.5 matches the classical free-free Bremsstrahlung 
run.radTheta = .0;

% With the whole X length of the grid taken as 1, these set the length of the equilibrium
% preshock & cold gas layers; Default values are .25 and .1 respectively
% Low-theta (~ <.2) shocks undergo large-amplitude breathing modes and require a large cold gas
% layer to let the large-amplitude downstream waves propagate away
run.fractionPreshock = 0.2;
run.fractionCold     = 0.4;

% This sets the temperature relative to the preshock temperature at which radiation rate is clamped
% to zero. Physically, the value makes little sense if < 1 (since the equilibrium generator assumes
% nonradiating preshock fluid). Practically, it must be slightly more than 1, or the radiation
% cutoff at the radiating-cold point becomes numerically unhappy. Default 1.05
%run.Tcutoff = 1.05

run.bcMode.x = ENUM.BCMODE_STATIC;

run.ppSave.dim2 = 100;
run.ppSave.dim3 = 01;

run.activeSlices.xy = false;
run.activeSlices.xyz = false;

run.saveFormat = ENUM.FORMAT_MAT;

run.seedAmplitude = 1e-1;

run.image.interval = 20;
run.image.mass = false;
%run.image.magY = true;
%run.image.pGas = true;

rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.firstCallIteration =1;
  rp.iterationsPerCall = 10;
  rp.spawnGUI = 1;
%run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLLC; 
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

