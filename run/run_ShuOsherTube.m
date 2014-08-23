% Run Shu Osher Tube test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 4 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = ShuOsherTubeInitializer(grid);
run.timeMax     = 0.178;
run.iterMax     = round(10*run.timeMax*grid(1)); % This will give steps max ~ 1.2x required

run.lambda 		= 8;
run.mach 		= 3;
run.waveAmplitude 	= .2;

run.alias       = '';
run.info        = 'Shu Osher Tube test.';

run.ppSave.dim2 = 5;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

