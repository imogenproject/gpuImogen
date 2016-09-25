%   Run 3D Corrugation instability shock test.

%--- Initialize test ---%
grid = [1024 16 1];
run         = CorrugationShockInitializer(grid);

run.iterMax     = 4000;
run.theta       = 0;
run.sonicMach   = 10;
run.alfvenMach  = .5;

run.bcMode.x = ENUM.BCMODE_CONSTANT;

run.ppSave.dim2 = 5;
run.ppSave.dim3 = 100;
run.seedAmplitude = 0e-6;


run.image.interval = 100;
run.image.mass = true;
%run.image.magY = true;
run.image.pGas = true;

%run.numericalICfile = sprintf('/data/Results/NASdata/shock_ini/IC_ms%i_ma0pt%2i_ang%i/3D_XYZ_FINAL.mat', run.sonicMach, round(100*run.alfvenMach), run.theta);

%run.radiation.type = ENUM.RADIATION_OPTICALLY_THIN;
%run.radiation.coolLength = .10;
%run.radiation.strengthMethod = 'coollen';

rez = run.geometry.globalDomainRez;
run.alias       = sprintf('IC_ms%i_ma0pt%2i_ang%i', run.sonicMach, round(100*run.alfvenMach), run.theta);
run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.alfvenMach, rez(1), rez(2), rez(3));

%--- Run tests ---%
if (true) %Primary test
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

