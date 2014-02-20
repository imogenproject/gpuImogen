% Run Sod shock tube test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 16 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = SodShockTubeInitializer(grid);
run.direction   = SodShockTubeInitializer.X;
run.shockAngle  = 0;
run.iterMax     = 5000;
run.timeMax     = 0.15;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 5;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

