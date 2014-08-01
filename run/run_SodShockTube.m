% Run Sod shock tube test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 8 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = SodShockTubeInitializer(grid);
run.direction   = SodShockTubeInitializer.X;
run.shockAngle  = 0;
run.timeMax     = 0.25;
run.iterMax     = 2*run.timeMax*grid(1); % This will give steps max ~ 1.2x required

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 5;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

