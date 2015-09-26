% Run 2D Orszag-Tang vortex test problem.

%--- Initialize test ---%
grid = [512 512 1];
run                 = OrszagTangVortexInitializer(grid);
run.info            = 'Orszag-Tang vortex: Resolution 1';
run.notes           = '';
run.profile         = false;
run.image.interval  = 50;
run.image.mass      = true;
run.image.parallelUniformColors = true;
run.iterMax = 10000;


%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

