% Run a test of the Kelvin-Helmholtz instability test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [512 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = KelvinHelmholtzInitializer(grid);
run.iterMax         = 100000;

run.mach = 2.0;
run.timeMax = (sqrt(5/3)/run.mach)*20;

run.direction       = KelvinHelmholtzInitializer.X;
run.image.interval  = 100;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

