% Run 2D Double Blast Wave test.

grid = [1024 1 1];
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
rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

