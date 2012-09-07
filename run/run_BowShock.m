%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize bow shock ---%
grid = [1024 1024 1];
run                 = BowShockInitializer(grid);
run.iterMax         = 2500;
%run.bcMode.z	    = 'circ';

run.bcMode.x = 'const';
run.bcMode.y = 'circ';
run.bcMode.z = 'circ';

run.cfl = .5;

run.gpuDeviceNumber = 0;

%--- Adjustable simulation parameters ---%

% Determine the part of the grid occupied by the obstacle
run.ballXRadius = 1;
run.ballCells = [63.5 63.5 63.5];
run.ballCenter =  [512 512 1];

% Nope, nope nope, nope...
run.mode.magnet = false;
run.magX = 0;
run.magY = 0;

% Set the parameters of the fluid the ball is embedded in
run.preshockRho = 1;
run.preshockP   = 1;
% And the mach of the incoming blastwave
run.blastMach   = 4;

% Set the parameters of the ball itself
run.ballRho = 1;
run.ballVr = 9;
run.ballXRadius = 1;
run.ballThermalPressure = 3;
run.ballLock = true;

%--- Adjustable output parameters ---%
run.image.interval  = 10;
run.image.mass      = true;
%run.image.speed     = true;
%run.image.pGas      = true;
%run.image.magX      = true;
%run.image.magY = true;

run.activeSlices.xy  = true;
%run.activeSlices.yz  = true;
%run.activeSlices.xyz = true;

run.ppSave.dim2     = 100;
%run.ppSave.dim3     = 20;


run.info            = 'Bow shock test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

