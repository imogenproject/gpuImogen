%   Run 3D Corrugation instability shock test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run         = CorrugationShockInitializer([512 4 1]);

run.iterMax     = 20000;
run.theta       = 60;
run.sonicMach   = 3;
run.alfvenMach  = .5;

run.cfl = 0.2;

%run.useGPU = true;
%run.gpuDeviceNumber = 0;
%run.bcMode.x = ENUM.BCMODE_CIRCULAR;

run.treadmill = 0;
run.dGrid.x = .01/4;

run.ppSave.dim2 = 25;
run.ppSave.dim3 = 100;
run.seedAmplitude = 0e-8;

run.image.interval = 100;
run.image.mass = true;

run.alias       = 'ms3_ma0pt50_ang60_ICGEN';
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
