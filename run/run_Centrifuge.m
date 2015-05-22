%--- Initialize test ---%
grid                = [3072 3072 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

% The centrifuge test provides an effective test for Imogen's rotating frame
% It calculates the equilibrium defined by a rotating-on-cylinders 2d fluid
% i.e. (drho/dr) (dP/drho) = - rho r w(r)^2
% solved for the case of an initially isothermal fluid with dP/drho == a^2/.
% The simulation then proceeds for an adiabatic fluid with index run.gamma
%
% This system is inextricably unstable (r^2 w decreases outward when the
% fluid comes to a stop at r=1) but because it is normally 2d, the axisymmetric
% overturn is prevented 
run                 = CentrifugeInitializer(grid);
run.iterMax         = 250000;

run.image.interval  = 100;
%run.image.speed     = true;
run.image.mass      = true;

run.activeSlices.xy = true;
%run.activeSlices.xz = true;
%run.activeSlices.xyz = false
run.ppSave.dim2 = 1;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CIRCULAR;

run.edgeFraction    = 2; % Sets the radius of the simulation to
% (1+this) times the size of the centrifuged region
run.gamma           = 5/3; % Sets the adiabatic index for fluid evolution
run.omega0          = 1; % Sets the w0 of w(r) = w0 (1-cos(2 pi r)) in the default rotation curve
run.rho0            = 1; % Sets the density at r >= 1 & the BC for the centrifuge region
run.P0              = 1;
run.minMass         = 1e-5; % enforced minimum density
run.frameRotateOmega= 0; % The rate at which the frame is rotating
run.eqnOfState      = EOS_ADIABATIC; % EOS_ISOTHERMAL or EOS_ADIABATIC or EOS_ISOCHORIC

run.pureHydro = true;
run.cfl = .45;

        run.useInSituAnalysis = 0;
        run.stepsPerInSitu = 10;
        run.inSituHandle = @RealtimePlotter;

run.info        = 'Testing centrifuged fluid equilibrium against rotating frame';
run.notes       = '';

run.image.parallelUniformColors = true;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

