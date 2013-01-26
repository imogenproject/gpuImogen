% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([14336 14336 1]);

%--- Initialize test ---%

run                 = OrszagTangVortexInitializer([14336 14336 1]);
run.info            = 'Orszag-Tang vortex test.';
run.notes           = '';
run.profile         = false;
run.image.interval  = 20;
run.image.mass      = true;
run.image.parallelUniformColors = true;
run.iterMax = 50000;

run.notes = 'Test that this turkey still flies in parallel and make some awesome pics';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

