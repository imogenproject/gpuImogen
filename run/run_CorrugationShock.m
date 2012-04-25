%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run         = CorrugationShockInitializer([256 4 1]);

run.iterMax     = 20000;
run.theta       = 30;
run.sonicMach   = 10;
run.alfvenMach  = .5;

run.gpuDeviceNumber = 0;
run.bcMode.x = ENUM.BCMODE_FADE;

run.ppSave.dim2 = .1;
run.ppSave.dim3 = 100;
run.seedAmplitude = 0e-8;

run.image.interval = 100;
%run.image.mass = true;
%run.image.magY = true;

%run.numericalICfile = sprintf('/data/Results/NASdata/shock_ini/IC_ms%i_ma0pt%2i_ang%i/3D_XYZ_FINAL.mat', run.sonicMach, round(100*run.alfvenMach), run.theta);

run.alias       = sprintf('IC_ms%i_ma0pt%2i_ang%i', run.sonicMach, round(100*run.alfvenMach), run.theta);
run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.alfvenMach, run.grid(1), run.grid(2), run.grid(3));
run.notes       = 'Experimenting with fade BC - ref run with const';

%--- Run tests ---%
if (true) %Primary test
    [mass, mom, ener, magnet, statics, ini] = run.getInitialConditions();
    IC.mass = mass;
    IC.mom = mom;
    IC.ener = ener;
    IC.magnet = magnet;
    IC.statics = statics;
    IC.ini = ini;
    icfile = [tempname '.mat'];

    save(icfile, 'IC');
    clear IC mass mom ener magnet statics ini run;
    imogen(icfile);
end

enderRun();
