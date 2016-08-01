
%   Simulate a circularly polarized alfen wave, which has an exact analytic solution.

%--- Initialize test ---%
grid = [256 128 128];
run         = CPAlfvenWave(grid);

%run.iterMax     = 2000;
%run.theta       = 0;
%run.sonicMach   = 5;
%run.alfvenMach  = .5;

%run.bcMode.x = ENUM.BCMODE_CONST;

%run.ppSave.dim2 = 5;
%run.ppSave.dim3 = 100;
%run.seedAmplitude = 100e-6;

%run.image.interval = 1;
%run.image.mass = true;
%run.image.magY = true;
%run.image.pGas = true;

%run.radiation.type = ENUM.RADIATION_OPTICALLY_THIN;
%run.radiation.coolLength = .10;
%run.radiation.strengthMethod = 'coollen';

run.alias = '';
%rez = run.geometry.globalDomainRez;
%run.alias       = sprintfs%i_ma0pt%2i_ang%i', run.sonicMach, round(100*run.alfvenMach), run.theta);
%run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
%                          run.theta, run.sonicMach, run.alfvenMach, rez(1), rez(2), rez(3));
%run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

enderRun();
