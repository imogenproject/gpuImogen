%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run         = CorrugationShockInitializer([4096 512 1]);

run.iterMax     = 30000;
run.theta       = 0;
run.sonicMach   = 3;
run.alfvenMach  = .5;

run.useGPU = true;
run.gpuDeviceNumber = 2;
run.bcMode.x = ENUM.BCMODE_CIRCULAR;

run.treadmill = 0;
run.dGrid.x = .01/512;

run.ppSave.dim2 = .2;
run.ppSave.dim3 = 100;

run.seedAmplitude = 1e-9;
run.randomSeed_spectrumLimit = 8;

run.bcInfinity = 50;

run.image.interval = 50;
run.image.mass = true;

run.numericalICfile = '/home/erik/group_data/NASdata/corr_2d/ICGEN_ms3_ma0pt50_ang0/3D_XYZ_FINAL.mat';

run.alias       = sprintf('CORR_ms%i_ma0pt%3i_ang%i',run.sonicMach, round(run.alfvenMach*1000),run.theta);
run.info        = sprintf('Corrugation instability test [Th=%g, Ms=%g, Ma=%g] with grid [%g, %g, %g]', ...
                          run.theta, run.sonicMach, run.alfvenMach, run.grid(1), run.grid(2), run.grid(3));
run.notes       = 'Toy run for purpose of calculating numerical initial conditions';

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
