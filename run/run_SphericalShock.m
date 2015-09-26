% Run the spherical shock test.

%--- Initialize test ---%
grid = [1366 2048 1];
run                 = SphericalShockInitializer(grid);
run.iterMax         = 15000;

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

