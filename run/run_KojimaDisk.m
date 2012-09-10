%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [400 400 100];
run                 = KojimaDiskInitializer(grid);
run.iterMax         = 200;
run.save            = true;
run.edgePadding     = 0.2;
run.pointRadius     = 0.25;
run.radiusRatio     = 0.67;
run.q = 1.75;

run.image.interval  = 5;
run.image.speed     = true;
run.image.mass      = true;

%run.specSaves.dim2  = [];
%run.ppSave.dim2     = 5;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;
run.ppSave.dim3 = 20;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CONST;

%run.addFade(ceil(grid/2), 16, ENUM.POINT_FADE , true, {ENUM.MOM});

run.pureHydro = true;
run.cfl = .8;

run.info        = 'Kojima disk simulation';
run.notes       = '';

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

