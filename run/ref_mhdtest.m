% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([1024 1024 1]);

%--- Initialize test ---%

run                 = OrszagTangVortexInitializer([1024 1024 1]);
run.info            = 'Orszag-Tang vortex test.';
run.notes           = '';
run.profile         = false;
run.image.interval  = 20;
run.image.mass      = true;
run.image.parallelUniformColors = true;
run.iterMax = 6000;

run.notes = 'Test that this turkey still flies with anticipated behavior.';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

