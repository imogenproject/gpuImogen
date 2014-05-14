%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [2048 2048 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

% The centrifuge test provides an effective test for Imogen's rotating frame
% It calculates the equilibrium defined by a rotating-on-cylinders 2d fluid
% i.e. (drho/dr) (dP/drho) = - rho r w(r)^2
% solved for the case of an initially isothermal fluid with dP/drho == a^2/.
run                 = CentrifugeInitializer(grid);
run.iterMax         = 10000;

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

obj.edgeFraction        = .5; % Sets the radius of the simulation to
% (1+this) times the size of the centrifuged region
obj.omega0              = 1; % Sets the w0 of w(r) = w0 (1-cos(2 pi r))
obj.rho0                = 1; % Sets the density at r >= 1
obj.a_isothermal        = 1; % Sets isothermal soundspeed ~ temperature
obj.minMass             = 1e-5; % enforced minimum density


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

