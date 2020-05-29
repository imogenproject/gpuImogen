%  Run a fluid jet test.

%--- Initialize test ---%
grid = [128 64 64];
run                 = JetInitializer(grid);
run.iterMax         = 5;
run.injectorSize    = [12 4];
run.injectorOffset          = [20 32 32];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.bcMode.z        = 'const'; % just for some variety
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 50;
run.image.mass      = true;
%run.image.speed     = true;

run.info            = 'Fluid jet test in 2D.';
run.notes           = '';

run.activeSlices.xyz = false;
run.ppSave.dim2 = 50;
run.ppSave.dim3 = 50;

run.jetMass = 3;
run.jetMach = 5;

run.pureHydro = 1;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.iterationsPerCall = 50;
  rp.firstCallIteration = 1;
  rp.forceRedraw=1;
  rp.spawnGUI = 1;
%run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

