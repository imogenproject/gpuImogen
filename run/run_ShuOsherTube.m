u% Run Shu Osher Tube test (Shock wave propagating through entropy wave)

%--- Initialize test ---%
grid = [1024 2 1];
run             = ShuOsherTubeInitializer(grid);
run.timeMax     = 0.178;
run.iterMax     = round(10*run.timeMax*grid(1)); % This will give steps max ~ 1.2x required

% These are the conditions per the original Shu & Osher paper:
run.lambda        = 8; % 8 waves in the box
run.mach          = 3; % Shock propagating at M=3
run.waveAmplitude = .2; % Preshock entropy fluctuation of density amplitude 0.2

run.alias       = 'SO_Tube';
run.info        = 'Shu Osher Tube test.';

run.ppSave.dim2 = 5;

rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 0;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
%run.peripherals{end+1} = rp;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToFile();
    outdir = imogen(IC);
end

