% Wave advection simulation

%grid = [512 512 1];
grid = [300 150 150];

%--- Initialize test ---%
run             = ShockBubbleInitializer(grid);
run.iterMax     = 10000;
run.timeMax = 0.25;
run.info        = 'Shock cloud test';
run.notes       = 'Smash a dense bubble into a shockwave!';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

run.ppSave.dim1 = 100;
if grid(3) == 1
    run.ppSave.dim2 = 10;
    run.ppSave.dim3 = 100;
else
    run.ppSave.dim2 = 100;
    run.ppSave.dim3 = 10;
end

run.bubbleDensity = 10;
run.bubbleRadius = .1;
run.shockMach = 6;

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.spawnGUI = 1;
  rp.iterationsPerCall = 10;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL; 
%  fm.toMethod = ENUM.CFD_HLLC;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
end

