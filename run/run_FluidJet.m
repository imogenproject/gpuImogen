%  Run a fluid jet test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run                 = JetInitializer([1024 512 1]);
run.iterMax         = 10000;
run.injectorSize    = 15;
run.offset          = [20 256 0];
run.bcMode.x        = 'const';
run.bcMode.y        = 'const';
run.bcMode.z        = 'circ'; % just for some variety
run.direction       = JetInitializer.X;
run.flip            = false;

run.image.interval  = 10;
run.image.mass      = true;
%run.image.speed     = true;

run.info            = 'Fluid jet test in 3D.';
run.notes           = '';

run.activeSlices.xyz = false;
run.ppSave.dim2 = 50;
run.ppSave.dim3 = 2;

run.injectorSize = 9;
run.jetMass = 3;
run.jetMach = 5;

run.pureHydro = 1;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

