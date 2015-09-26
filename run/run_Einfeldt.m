% Run Einfeldt Strong Rarefaction test.

%--- Initialize test ---%
grid = [1024 2 1];
run             = EinfeldtInitializer(grid);
run.timeMax     = 0.1;
run.iterMax     = 2*run.timeMax*grid(1); % This will give steps max ~ 1.2x required

run.cfl         = .4;

run.rhol        = 1;
run.ml          = -1;
run.nl          = 0;
run.el          = 5;

run.rhor        = 1;
run.mr          = 1;
run.nr          = 0;
run.er          = 5;

% Note that with HLL, a Delta m of greater than 2.1 results in too great of a rarefaction, resulting in negative mass density and NAN-Plague.
% With HLLC, this value is between 2.6 and 2.8
run.alias       = '';
run.info        = 'Einfeldt Strong Rarefaction test.';
run.notes        = '';
run.ppSave.dim2 = 5;

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 20;
run.inSituHandle = @RealtimePlotter;
        instruct.plotmode = 1;
        instruct.plotDifference = 0;
        instruct.pause = 0;
run.inSituInstructions = instruct;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

