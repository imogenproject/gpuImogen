% Run a Sedov Taylor blast wave test.

%-- Initialize Imogen directory ---%
starterRun();
grid = [128 128 128];

%--- Initialize test ---%
run         = SedovTaylorBlastWaveInitializer(grid);

run.autoEndtime = 1; % Automatically run until R = 0.45
run.iterMax = 10000;

run.alias   = '';
run.info    = 'Sedov-Taylor blast wave test.';
run.notes   = 'Just a test...';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

