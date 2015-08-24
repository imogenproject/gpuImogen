
% Run a Sedov Taylor blast wave test.

%-- Initialize Imogen directory ---%
grid = [192 192 192];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run         = SedovTaylorBlastWaveInitializer(grid);
run.iterMax = 5000; % It only really matter that this > 2*resolution to be safe

run.autoEndtime = 1;
run.depositRadiusCells(2.5);

run.ppSave.dim3 = 10;

run.alias   = '';
run.info    = 'Sedov-Taylor blast wave test.';
run.notes   = 'Just a test...';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

