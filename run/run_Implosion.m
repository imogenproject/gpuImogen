% Run the implosion symmetry test.

%--- Initialize test ---%
grid = [512 512 1];
run                 = ImplosionInitializer(grid);
run.iterMax         = 2500;

obj.Mcorner         = 0.125;
obj.Pcorner         = 0.14;

run.image.interval  = 100;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Implosion symmetry test';
run.notes           = '';
run.ppSave.dim2     = 10;

% The following tracks the asymmetry of the implosion over time

rp = RealtimePlotter();
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
%run.peripherals{end+1} = rp;
ma = MassConservationAnalyzer();
  ma.stepsPerCheck = 1;
  ma.plotResult = 1;
run.peripherals{end+1} = ma;


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

