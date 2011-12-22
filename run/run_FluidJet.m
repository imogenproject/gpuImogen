%  Run a fluid jet test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run                 = JetInitializer([256 192 192]);
run.iterMax         = 500;
run.injectorSize    = 15;
run.offset          = [10 96 96];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.bcMode.z        = 'circ'; % just for some variety
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 15;
run.image.mass      = true;
%run.image.speed     = true;

run.info            = 'Fluid jet test in 3D.';
run.notes           = '';

run.activeSlices.xyz = true;
run.ppSave.dim2 = 50;
run.ppSave.dim3 = 2;

run.injectorSize = 9;
run.jetMass = 3;
run.jetMach = 5;

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
