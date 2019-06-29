
%--- Initialize bow shock ---%
grid = [512 512 1];
run                 = BowShockInitializer(grid);
run.iterMax         = 2000;
%run.bcMode.z            = 'circ';

run.bcMode.x = 'const';
run.bcMode.y = 'circ';
run.bcMode.z = 'circ';

run.cfl = .85;

%--- Adjustable simulation parameters ---%

% This sets the radiation prefactor in the parameterized equation
%        \Gamma = -beta rho^2 T^theta
% It's irrelevant outside of changing output units because this parameter has a fixed relation 
% to the cooling length and the simulation automatically re-scales dx based on the 
% fractionPreshock and fractionCold parameters.
run.radBeta = 1;

% Sets the temperature dependence of the cooling equation
% theta = 0.5 matches the classical free-free Bremsstrahlung 
run.radTheta = -1;

% Determine the part of the grid occupied by the obstacle
run.ballXRadius = 1;
run.ballCells = [1 1 1]*round(grid(1)/8) +.5;
run.ballCenter =  ceil(grid/2);

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
run.ballVr = 4;
run.ballXRadius = 1;
run.ballThermalPressure = 1;
run.ballLock = true;

%--- Adjustable output parameters ---%
%run.image.interval  = 20;
%run.image.mass      = true;
%run.image.speed     = true;
%run.image.pGas      = true;
%run.image.magX      = true;
%run.image.magY = true;

run.activeSlices.xy  = true;
%run.activeSlices.yz  = true;
%run.activeSlices.xyz = true;

run.ppSave.dim2     = 100;
%run.ppSave.dim3     = 20;

rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
  rp.spawnGUI = 1;
run.peripherals{end+1} = rp;

run.info            = 'Bow shock test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end
