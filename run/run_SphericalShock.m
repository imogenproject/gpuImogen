% Run the spherical shock test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1366 2048 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = SphericalShockInitializer(grid);
run.iterMax         = 15000;

run.direction       = SphericalShockInitializer.X;
run.image.interval  = 20;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Spherical Shock test';
run.notes           = '';


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

