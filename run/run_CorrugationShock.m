%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([1024 128 1]);

%--- Initialize test ---%
run         = CorrugationShockInitializer([1024 128 1]);

run.iterMax     = 20000;
run.theta       = 0;
run.sonicMach   = 5;
run.alfvenMach  = .5;

run.bcMode.x = ENUM.BCMODE_FADE;

run.ppSave.dim2 = 5;
run.ppSave.dim3 = 100;
run.seedAmplitude = 100e-8;

run.image.interval = 50;
run.image.mass = true;
%run.image.magY = true;
run.image.pGas = true;

run.numericalICfile = sprintf('/data/Results/NASdata/shock_ini/IC_ms%i_ma0pt%2i_ang%i/3D_XYZ_FINAL.mat', run.sonicMach, round(100*run.alfvenMach), run.theta);

%run.radiation.type = ENUM.RADIATION_OPTICALLY_THIN;
%run.radiation.coolLength = .10;
%run.radiation.strengthMethod = 'coollen';

run.alias       = sprintf('IC_ms%i_ma0pt%2i_ang%i', run.sonicMach, round(100*run.alfvenMach), run.theta);
run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.alfvenMach, run.grid(1), run.grid(2), run.grid(3));
%run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

