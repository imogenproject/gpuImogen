% Run Sod shock tube test.

%--- Initialize test ---%
grid = [1024 2 1];
run             = SodShockTubeInitializer(grid);
run.normal([1 0 0]);

run.cfl = .4;
run.timeMax     = 0.2;
run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.bcMode.x = ENUM.BCMODE_CONST;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 20;
run.inSituHandle = @RealtimePlotter;
        instruct.plotmode = 1;
        instruct.plotDifference = 0;
        instruct.pause = 0;
run.inSituInstructions = instruct;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

