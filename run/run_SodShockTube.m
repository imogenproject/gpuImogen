% Run Sod shock tube test.

%--- Initialize test ---%
grid = [12800 1 1];
run             = RiemannProblemInitializer(grid);

run.demo_SodTube();

run.cfl = .4;
run.iterMax     = 3*run.timeMax*grid(1)/run.cfl; % This will give steps max ~ 1.2x required

run.bcMode = ENUM.BCMODE_CONSTANT;

run.alias       = '';
run.info        = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim3 = 100;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
%run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 2; % hllc
run.peripherals{end+1} = fm;

run.saveFormat = ENUM.FORMAT_MAT;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

