
%--- Initialize bow shock ---%
grid = [96 96 96];
run                 = BowShockInitializer(grid);
run.iterMax         = 400;
%run.bcMode.z            = 'circ';

run.bcMode.x = {ENUM.BCMODE_STATIC, ENUM.BCMODE_CONSTANT};
run.bcMode.y = ENUM.BCMODE_CONSTANT;
run.bcMode.z = ENUM.BCMODE_CONSTANT;

run.cfl = .85;

run.ymirrorsym = 1;
run.zmirrorsym = 1;

%--- Adjustable simulation parameters ---%

% This sets the radiation prefactor in the parameterized equation
%        \Gamma = -beta rho^2 T^theta
% It's irrelevant outside of changing output units because this parameter has a fixed relation 
% to the cooling length and the simulation automatically re-scales dx based on the 
% fractionPreshock and fractionCold parameters.
run.radBeta = 1;

% Sets the temperature dependence of the cooling equation
% theta = 0.5 matches the classical free-free Bremsstrahlung 
run.radTheta = -.5;

run.radCoollen=.3;

% Determine the part of the grid occupied by the obstacle
run.ballXRadius = 1;
run.ballCells = [1 1 1]*round(grid(1)/8) +.5;
run.ballCenter =  ceil([grid(1)*.7 grid(2)/2 grid(3)/2]);

% Nope, nope nope, nope...
run.mode.magnet = false;
run.magX = 0;
run.magY = 0;

% Set the parameters of the fluid the ball is embedded in
run.preshockRho = 1;
run.preshockP   = 1;
% And the mach of the incoming blastwave
run.blastMach   = 6;

% Set the parameters of the ball itself
run.ballRho = 4;
run.ballVr = 11;
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

run.activeSlices.xy  = false;
%run.activeSlices.yz  = true;
run.activeSlices.xyz = true;

run.ppSave.dim2     = 100;
run.ppSave.dim3     = 10;

rp = RealtimePlotter();
  rp.plotmode = 1;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.firstCallIteration = 1;
  rp.iterationsPerCall = 20;
  rp.spawnGUI = 1;

  rp.plotmode = 1;
rp.cut = round(grid/2);
rp.indSubs = [1 1 128;1 1 256;1 1 256];
rp.movieProps(0, 0, 'RTP_');
rp.vectorToPlotprops(1, [1  10   0   4   1   0   0   0   0   1  10   1   8   1   0   0   0]);

  
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL; 
run.peripherals{end+1}=fm;

run.info            = 'Bow shock test.';
run.notes           = '';

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

