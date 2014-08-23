% Run 2D Double Blast Wave test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [1024 8 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = DoubleBlastInitializer(grid);
run.timeMax     = 0.038;
run.iterMax     = 5000;%2*run.timeMax*grid(1); % This will give steps max ~ 1.2x required

run.alias       = '';
run.info        = '2D Double Blast Wave test.';
run.notes       = '';

run.pr		= 100;
run.pl		= 1000;
run.pa		= .01;

run.ppSave.dim2 = 5;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

