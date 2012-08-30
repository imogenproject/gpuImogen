%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [256 256 128];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 100;
run.save            = true;
run.edgePadding     = 0.2;
run.pointRadius     = 0.25;
run.radiusRatio     = 0.67;
run.q = 1.75;

run.image.interval  = 10;
run.image.speed     = true;
run.image.mass      = true;

run.specSaves.dim2  = [];
run.ppSave.dim2     = 10;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CONST;

%run.addFade(ceil(grid/2), 16, ENUM.POINT_FADE , true, {ENUM.MOM});

run.gpuDeviceNumber = 0;
run.pureHydro = true;

run.info        = 'Kojima disk simulation';
run.notes       = '';

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

