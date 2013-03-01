%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid                = [256 256 192];

GIS = GlobalIndexSemantics(); GIS.setup(grid);

run                 = KojimaDiskInitializer(grid);
run.iterMax         = 20;
run.edgePadding     = 0.2;
run.pointRadius     = 0.15;
run.radiusRatio     = 0.70;
run.q = 2.1;

run.image.interval  = 2;
%run.image.speed     = true;
run.image.mass      = true;

run.activeSlices.xy = false;
%run.activeSlices.xz = true;
run.activeSlices.xyz = false;
run.ppSave.dim3 = 50;

run.bcMode.x        = ENUM.BCMODE_CONST;
run.bcMode.y        = ENUM.BCMODE_CONST;
run.bcMode.z        = ENUM.BCMODE_CONST;

run.pureHydro = true;
run.cfl = .75;

run.info        = 'Kojima disk simulation';
run.notes       = '';

run.image.parallelUniformColors = true;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

