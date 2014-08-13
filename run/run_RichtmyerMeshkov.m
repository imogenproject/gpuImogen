% Run the Richtmyer-Meshkov instability test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [2048 2048 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = RichtmyerMeshkovInitializer(grid);
run.iterMax         = 20000;

run.direction       = RichtmyerMeshkovInitializer.X;
run.image.interval  = 75;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Richtmyer-Meshkov instability test';
run.notes           = '';


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

