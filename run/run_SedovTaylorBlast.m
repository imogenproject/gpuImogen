% Run a Sedov Taylor blast wave test.
% Imogen's ST analyzer is capable of handling 1- 2- and 3-D explosions
% with our uniform background case

%-- Initialize Imogen directory ---%
grid = [128 128 128];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run         = SedovTaylorBlastWaveInitializer(grid);
% There is a nominally easy way to calculate what this should be
% but I don't have it on hand. Just put in a large number...
run.iterMax = 50000;

% If set to 1, this sets run.timeMax such that the explosion will
% span 90% of the grid's diameter
run.autoEndtime = 1;

% One method of initializing the center of the blast: Deposit the energy equally
% into cells whose radius is less than this
run.depositRadiusCells(2.5);

run.ppSave.dim3 = 10;

run.alias   = '';
run.info    = 'Sedov-Taylor blast wave test.';
run.notes   = 'Just a test...';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    outdir = imogen(icfile);
% FIXME: This should be implemented as an in-situ analyzer to avoid massive I/O if actual
% examination of output data is not desired.
    howdo = analyzeSedovTaylor(outdir, 1);

    if mpi_amirank0()
        disp(howdo)
    end
end

