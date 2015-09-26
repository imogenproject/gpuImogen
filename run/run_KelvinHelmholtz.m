k% Run a test of the Kelvin-Helmholtz instability.

%--- Initialize test ---%
grid = [512 512 1];
run                 = KelvinHelmholtzInitializer(grid);
run.iterMax         = 1000;

run.waveHeight      = .01;         % Creates sine waves at the boundaries of this amplitude (fraction of total grid)
run.numWave         = 5;         % Sine waves have this many peaks
run.randAmp         = .01;         % X and Y velocity is given random components of this amplitude
run.mach            = 2.0;        % Each region is given this speed, so total dv at the boundary is twice as large as this.
run.timeMax         = (sqrt(5/3)/run.mach)*20;

run.image.interval  = 50;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Kelvin-Helmholtz instability test.';
run.notes           = '';

run.useInSituAnalysis = 0;
run.stepsPerInSitu = 20;
run.inSituHandle = @RealtimePlotter;
        instruct.plotmode = 4;
        instruct.plotDifference = 0;
        instruct.pause = 0;
run.inSituInstructions = instruct;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

