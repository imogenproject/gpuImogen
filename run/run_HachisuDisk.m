% Simulate the evolution of a Hachisu self-gravitating rotates-on-cylinders equilibrium

%--- Initialize test ---%
%        The number of X cells is the controlling variable in grid step size.
grid = [192 192 128];
run                 = HachisuDiskInitializer(grid);
run.iterMax         = 200;

run.q = 2;
%run.d
run.aspectRatio = .15;

run.gravity.constant = 1;

run.image.interval   = 1;
run.image.mass       = true;
run.image.speed      = true;
run.image.momZ       = true;
%run.image.ener       = true;

run.activeSlices.yz  = true;
run.activeSlices.xyz = true;

run.ppSave.dim3 = 10; % every 10 steps
%run.ppSave.dim2 = 10;

run.info            = 'Toy gravity problem test';
run.notes           = '';

% lol broken
run.gravity.solver = 'biconj_gpu';
run.gravity.tolerance = 1e-10;
run.gravity.iterMax = 150;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

