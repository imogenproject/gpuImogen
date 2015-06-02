% Run Sod shock tube test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
grid = [8192 2 1];
run             = SodShockTubeInitializer(grid);
run.normal([1 0 0]);

run.cfl = .4;
run.timeMax     = 0.2;
run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.bcMode.x = ENUM.BCMODE_CONST;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 12.5;

        run.useInSituAnalysis = 1;
        run.stepsPerInSitu = 20;
        run.inSituHandle = @RealtimePlotter;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

