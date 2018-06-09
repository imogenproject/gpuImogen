% Generate some von Karman vortices

%--- Initialize test ---%
grid = [1024 512 1];
run                 = KarmanStreetInitializer(grid);
run.iterMax         = 2000;

run.image.interval  = 25;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Karman Street test';
run.notes           = '';
run.ppSave.dim2     = 25;
run.mach            = .8;

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
%run.peripherals{end+1} = rp;


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

