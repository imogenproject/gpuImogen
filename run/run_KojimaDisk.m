%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [1024 1024 1];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 1000;
run.save            = true;
run.edgePadding     = 0.2;
run.pointRadius     = 0.25;
run.radiusRatio     = 0.67;
run.q = 1.75;

run.image.interval  = 20;
run.image.speed     = true;
run.image.mass      = true;

run.specSaves.dim2  = [];
run.ppSave.dim2     = 5;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;

%run.addFade(ceil(grid/2), 16, ENUM.POINT_FADE , true, {ENUM.MOM});

run.useGPU = true;
run.gpuDeviceNumber = 2;
run.pureHydro = true;

run.info        = 'Kojima disk simulation';
run.notes       = '';

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
