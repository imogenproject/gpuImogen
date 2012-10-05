% Run 2D Orszag-Tang vortex test problem.

%-- Initialize Imogen directory ---%
starterRun();
GIS = GlobalIndexSemantics(); GIS.setup([1042 1024 1]);

%--- Initialize test ---%
run                 = OrszagTangVortexInitializer([1042 1024 1]);
run.info            = 'Orszag-Tang vortex test.';
run.notes           = '';
run.profile         = false;
run.image.interval  = 50;
run.image.mass      = true;
run.iterMax = 20000;

%run.notes = 'Testing new magnetic flux ordering';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

