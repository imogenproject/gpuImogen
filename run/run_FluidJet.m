%  Run a fluid jet test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run                 = JetInitializer([768 512 1]);
run.iterMax         = 4000;
run.injectorSize    = 15;
run.offset          = [40 256 1];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 5;
run.image.mass      = true;
run.image.speed     = true;

run.info            = 'Fluid jet test.';
run.notes           = '';

run.injectorSize = 20;
run.jetMass = 1.8;
run.jetMach = 4;

run.useGPU = true;
run.gpuDeviceNumber = 2;
run.pureHydro = 1;

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
