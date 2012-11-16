% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([16384 16384 1]);

%--- Initialize test ---%

run.bcModes.x = 'circ';
run.bcModes.y = 'circ';

run                 = OrszagTangVortexInitializer([16384 16384 1]);
run.info            = 'Orszag-Tang vortex test.';
run.notes           = '';
run.profile         = false;
run.image.interval  = 50;
run.image.mass      = true;
run.iterMax = 99999;

%run.notes = 'Testing new magnetic flux ordering';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

