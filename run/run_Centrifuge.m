%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [512 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

% The centrifuge test provides an effective test for Imogen's rotating frame
% It calculates the equilibrium defined by a rotating-on-cylinders 2d fluid
% i.e. (drho/dr) (dP/drho) = - rho r w(r)^2
% solved for the case of an initially isothermal fluid with dP/drho == a^2/.
% The simulation then proceeds for an adiabatic fluid with index run.gamma
%
% This system is inextricably unstable (r^2 w decreases outward when the
% fluid comes to a stop at r=1) and will show nonaxisymmetric instabilities
% if run long enough. At 512^2 these reach visible amplitude ~4000 iterations
run                 = CentrifugeInitializer(grid);
run.iterMax         = 2000;

run.image.interval  = 20;
%run.image.speed     = true;
run.image.mass      = true;

run.activeSlices.xy = true;
%run.activeSlices.xz = true;
%run.activeSlices.xyz = false
run.ppSave.dim2 = 1;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CONST;

run.edgeFraction    = .5; % Sets the radius of the simulation to
% (1+this) times the size of the centrifuged region
run.gamma           = 5/3; % Sets the adiabatic index for fluid evolution
run.omega0          = 1; % Sets the w0 of w(r) = w0 (1-cos(2 pi r)) in the default rotation curve
run.rho0            = 1; % Sets the density at r >= 1 & the BC for the centrifuge region
run.cs0             = 1; % Sets soundspeed, may be interpreted as isothermal or adiabatic depending on EoS
run.polyK           = 1; % Sets k in P = k rho^gamma if using initial adiabatic EoS
run.minMass         = 1e-5; % enforced minimum density
run.frameRotateOmega = 1; % The rate at which the frame is rotating
run.eqnOfState      = run.EOS_ADIABATIC; % or EOS_ADIABATIC or EOS_ISODENSITY

run.pureHydro = true;
run.cfl = .75;

run.info        = 'Testing centrifuged fluid equilibrium against rotating frame';
run.notes       = '';

run.image.parallelUniformColors = true;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

