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

rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 10;
run.peripherals{end+1} = rp;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

