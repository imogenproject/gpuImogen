% Run a test of the Kelvin-Helmholtz instability test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 1024 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run                 = KelvinHelmholtzInitializer(grid);
run.iterMax         = 2000;
run.direction       = KelvinHelmholtzInitializer.X;
run.image.interval  = 25;
run.image.mass      = true;
run.image.mach      = true;
run.activeSlices.xy = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

