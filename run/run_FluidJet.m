%  Run a fluid jet test.

%--- Initialize test ---%
grid = [1024 512 1];
run                 = JetInitializer(grid);
run.iterMax         = 1000;
run.injectorSize    = 15;
run.offset          = [20 256 0];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.bcMode.z        = 'circ'; % just for some variety
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 10;
run.image.mass      = true;
%run.image.speed     = true;

run.info            = 'Fluid jet test in 2D.';
run.notes           = '';

run.activeSlices.xyz = false;
run.ppSave.dim2 = 50;
run.ppSave.dim3 = 50;

run.injectorSize = 9;
run.jetMass = 3;
run.jetMach = 5;

run.pureHydro = 1;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
%run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = 3; % xin/jin
run.peripherals{end+1} = fm;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

