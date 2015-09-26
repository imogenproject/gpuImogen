% Run the Richtmyer-Meshkov instability test.

%--- Initialize test ---%
grid = [512 512 1];
run                 = RichtmyerMeshkovInitializer(grid);
run.iterMax         = 700;

run.numWave         = 1;
run.waveHeight      = 1/20;
run.image.interval  = 50;
run.image.mass      = true;
run.activeSlices.xy = true;
run.info            = 'Richtmyer-Meshkov instability test';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

