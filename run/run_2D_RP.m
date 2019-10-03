grid = [512 512 1];
run         = RiemannProblemInitializer(grid);
run.iterMax = 10000;

%run.demo_interactingRP1();
run.demo_interactingRP2();

run.bcMode.x    = ENUM.BCMODE_CONSTANT;
run.bcMode.y    = ENUM.BCMODE_CONSTANT;

run.alias       = 'RP';
run.info        = '2D Riemann test';
run.notes        = '';
run.ppSave.dim2 = 10;

% TEST DEMO
% Instructs the simulation to start with HLLC, then hop to HLL after 20 steps
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL; % hll
  fm.toMethod = ENUM.CFD_HLL; % hll. hll = 1
  fm.atstep = -1;
run.peripherals{end+1} = fm;
rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.spawnGUI = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 1;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;


%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

