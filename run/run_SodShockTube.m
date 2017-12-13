% Run Sod shock tube test.

%--- Initialize test ---%
grid = [2560 1 1];
run             = RiemannProblemInitializer(grid);

run.demo_SodTube();

run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.bcMode = ENUM.BCMODE_CONSTANT;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
  rp.spawnGUI = 1;
  rp.forceRedraw = 1;
%run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLLC;
run.peripherals{end+1} = fm;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
    icfile = run.saveInitialCondsToFile();
    outpath = imogen(icfile);


