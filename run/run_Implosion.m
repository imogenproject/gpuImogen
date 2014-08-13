% Run the implosion symmetry test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [2048 2048 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = ImplosionInitializer(grid);
run.iterMax         = 30000;

run.direction       = ImplosionInitializer.X;
run.image.interval  = 50;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Implosion symmetry test';
run.notes           = '';


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

