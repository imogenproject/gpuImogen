% Run a test of the Kelvin-Helmholtz instability.

%--- Initialize test ---%
grid = [512 512 1];
run                 = KelvinHelmholtzInitializer(grid);
run.iterMax         = 1000;

run.waveHeight      = .01;       % Creates sine waves at the boundaries of this amplitude (as fraction of total grid)
run.numWave         = 5;         % Sine waves have this many peaks
run.randAmp         = .01;       % X and Y velocity is given random components of this amplitude
run.mach            = 2.0;       % Each region is given this speed, so total velocity jump at the boundary is twice as large as this.
run.timeMax         = (sqrt(5/3)/run.mach)*20;

run.image.interval  = 50;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

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

