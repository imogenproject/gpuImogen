%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run         = CorrugationShockInitializer([24576 512 1]);

run.iterMax     = 100000;
run.theta       = 0;
run.sonicMach   = 4;
run.alfvenMach  = .9;

run.gpuDeviceNumber = 2;
run.bcMode.x = ENUM.BCMODE_CONST;

run.ppSave.dim2 = 1;
run.ppSave.dim3 = 100;
run.seedAmplitude = 1e-6;

run.image.interval = 100;
run.image.mass = true;
run.image.magY = true;

run.numericalICfile = '/data/Results/NASdata/corr_2d/ICGEN_ms4_ma0pt90_ang0/3D_XYZ_FINAL.mat';

run.alias       = 'ms4_ma0pt9_ang0';
run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.alfvenMach, run.grid(1), run.grid(2), run.grid(3));
run.notes       = 'Corrugation instability test with maximal transverse resolution yet';

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
