%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();
grid = [3072 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run         = RadiatingShockInitializer(grid);

run.iterMax     = 2;
run.theta       = 0;
run.sonicMach   = 4;

run.radBeta = 1;
run.radTheta = 1;

run.bcMode.x = ENUM.BCMODE_CONST;

run.ppSave.dim2 = .4;
run.ppSave.dim3 = 100;
run.seedAmplitude = 1e-2;

run.image.interval = 20;
run.image.mass = true;
%run.image.magY = true;
%run.image.pGas = true;

%run.numericalICfile = sprintf('/data/Results/NASdata/shock_ini/IC_ms%i_ma0pt%2i_ang%i/3D_XYZ_FINAL.mat', run.sonicMach, round(100*run.alfvenMach), run.theta);

%run.radiation.type = ENUM.RADIATION_OPTICALLY_THIN;
%run.radiation.coolLength = .10;
%run.radiation.strengthMethod = 'coollen';

run.alias       = sprintf('RHD_ms%i_ang%i', run.sonicMach, run.theta);
run.info        = sprintf('Radiating hydrodynamic shock test [Th=%g, Ms=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.grid(1), run.grid(2), run.grid(3));
%run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

