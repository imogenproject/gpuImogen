% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([600 600 1]);

%--- Initialize test ---%

run                 = OrszagTangVortexInitializer([600 600 1]);
run.info            = 'Orszag-Tang vortex test.';
run.notes           = '';
run.profile         = false;
run.image.interval  = 20;
run.image.mass      = true;
run.image.parallelUniformColors = true;
run.iterMax = 9999;

%run.notes = 'Testing new magnetic flux ordering';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

