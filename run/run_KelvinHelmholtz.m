% Run a test of the Kelvin-Helmholtz instability.

%--- Initialize test ---%
grid = [384 768 1];
run                 = KelvinHelmholtzInitializer(grid);
run.iterMax         = 100000;

run.waveHeight      = .05;       % Creates sine waves at the boundaries of this amplitude (as fraction of total grid)
run.nx         = 1;         % Sine waves have this many peaks
run.mach            = .20;       % Each region is given this speed, so total velocity jump at the boundary is twice as large as this.
run.massRatio = 1;
run.timeMax         = (sqrt(5/3)/run.mach)*200;

run.image.interval  = 50;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

rp = RealtimePlotter();
rp.spawnGUI = 1;
  rp.plotmode = 4;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.iterationsPerCall = 20;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

