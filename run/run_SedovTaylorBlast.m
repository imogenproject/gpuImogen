% Run a Sedov Taylor blast wave test.

%-- Initialize Imogen directory ---%
starterRun();
grid = [65 65 65];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run         = SedovTaylorBlastWaveInitializer([65 65 65]);
run.iterMax = 100;

run.alias   = '';
run.info    = 'Sedov-Taylor blast wave test.';
run.notes   = 'Just a test...';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

