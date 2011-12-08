% Run Sod shock tube test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run             = SodShockTubeInitializer([1024 1024 1]);
run.direction   = SodShockTubeInitializer.X;
run.shockAngle  = 30;
run.iterMax     = 1000;
run.timeMax     = 0.15;

run.useGPU = true;
run.gpuDeviceNumber = 2;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = '1024 resolution non-axis-aligned Sod shock tube test in the XY direction (theta = 30 degrees).';

run.ppSave.dim2 = 10;
run.bcMode.x = 'circ';
run.bcMode.y = 'circ';
run.bcMode.z = 'circ';

%--- Run tests ---%
if (true)
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
