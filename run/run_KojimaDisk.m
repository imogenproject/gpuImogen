%   Run a Kojima disk model.

%--- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
<<<<<<< HEAD
grid                = [768 768 384];
=======
grid                = [200 200 400];
>>>>>>> 52856bd54b13eb0e736aa2a3c7071515ccd607cf

GIS = GlobalIndexSemantics(); GIS.setup(grid);

run                 = KojimaDiskInitializer(grid);
<<<<<<< HEAD
run.iterMax         = 40000;
=======
run.iterMax         = 20;
>>>>>>> 52856bd54b13eb0e736aa2a3c7071515ccd607cf
run.edgePadding     = 0.2;
run.pointRadius     = 0.20;
run.radiusRatio     = 0.65;
run.q = 2.2;

run.image.interval  = 50;
%run.image.speed     = true;
run.image.mass      = true;

run.activeSlices.xy = true;
%run.activeSlices.xz = true;
run.activeSlices.xyz = true
run.ppSave.dim3 = .5;

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

