% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
grid = [1024 1024 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%

run                 = OrszagTangVortexInitializer(grid);
run.info            = 'Orszag-Tang vortex: Resolution 1';
run.notes           = '';
run.profile         = false;
run.image.interval  = 50;
run.image.mass      = true;
run.image.parallelUniformColors = true;
run.iterMax = 10000;

run.notes = 'Test that this turkey still flies in parallel and make some awesome pics';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

