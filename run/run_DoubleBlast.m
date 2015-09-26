% Run 2D Double Blast Wave test.

grid = [1024 2 1];
%--- Initialize test ---%
run             = DoubleBlastInitializer(grid);
run.timeMax     = 0.038;
run.iterMax     = 5000;

run.alias       = '';
run.info        = '2D Double Blast Wave test.';
run.notes       = '';

run.pRight      = 100;
run.pLeft       = 1000;
run.pMid        = .01;

run.ppSave.dim2 = 5;

% Generate realtime output of simulation results.
        run.useInSituAnalysis = 0;
        run.stepsPerInSitu = 20;
        run.inSituHandle = @RealtimePlotter;
instruct.plotmode = 1;
instruct.plotDifference = 0;
        run.inSituInstructions = instruct;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

