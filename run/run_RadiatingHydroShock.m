%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
grid = [512 2 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run         = RadiatingShockInitializer(grid);

run.iterMax     = 20;
run.theta       = 0;
run.sonicMach   = 4;

% This sets the radiation prefactor in the parameterized equation
%	\Gamma = -beta rho^2 T^theta
% It's irrelevant outside of changing output units because this parameter has a fixed relation 
% to the cooling length and the simulation automatically renormalizes dx based on the 
% fractionPreshock and fractionCold parameters.
run.radBeta = 1;

% Sets the temperature dependence of the cooling equation
% theta = 0.5 matches the classical fre-free Bremsstrahlung 
run.radTheta = .1;

% With the whole X length of the grid taken as 1, these set the length of the equilibrium
% preshock & cold gas layers; Default values are .25 and .1 respectively
% Low-theta (~ <.2) shocks undergo large-amplitude breathing modes and require a large cold gas
% layer to let the large-amplitude downstream waves propagate away
%run.fractionPreshock = 0.25;
%run.fractionCold     = 0.1;

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

run.useInSituAnalysis = 1;
run.stepsPerInSitu = 5;
	instruct.plotmode = 1;
	instruct.plotDifference = 1;
	instruct.pause = 0;
run.inSituInstructions = instruct;
run.inSituHandle = @RealtimePlotter;

run.alias       = sprintf('RHD_ms%i_ang%i', run.sonicMach, run.theta);
run.info        = sprintf('Radiating hydrodynamic shock test [Th=%g, Ms=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.grid(1), run.grid(2), run.grid(3));
%run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

