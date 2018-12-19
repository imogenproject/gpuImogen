%   Run a Kojima disk model.

grid                = [512 32 128];

run                 = ShearingBoxInitializer(grid);
run.iterMax         = 1000;

run.Mstar           = 2e30;
run.Rstar           = 1e9;

% Physical input parameters:
% - innerRadius, outerRadius: obvious
% - Mstar, Rstar: Mass & radius of star (radius not meaningful yet)
% - Sigma0: surface mass density at r0
% - densityExponent: sigma = sigma_0 (r/r0)^densityExponent
% - dustFraction: rho_dust = dustFraction * rho_gas
% - cs0: isothermal sound speed (= kb T / mu) at r=r0
% - temperatureExponent: c_s(r) = cs0 * (r/r0)^temperatureExponent

% M=6 will simulate a 60* wedge, avoiding the requirement of a really excessive number of
% phi direction cells, if the inability to resolve truly global (m=1) structure is OK
run.azimuthalMode = 6;

% Numeric parameters:
% - densityCutoffFraction: stop density falloff below rho = f*rho_max
% - useZMirror: simulate only upper half of disk, Z- BC automatically set to mirror
% - normalizationRadius - r0
% - normalizeValues - 

run.normalizationRadius = 150e10; % 10AU
run.normalizeValues = 1;


% 5 to 15 AU
run.innerRadius = 150e9*5;
run.outerRadius = 150e9*20;

run.Sigma0 = 1000; % 100g/cm^2 = 1000kg/m^2
run.densityExponent = -1; % steady state accretion disk
run.temperatureExponent = -0.5; % P_solar ~ r^-2 = P_bbody ~ T^4 -> T ~ r^-0.5

run.dustFraction = .01; % just try to make a gas disk work...

run.gasPerturb = .01; % disturb at 1% soundspeed

run.useZMirror = 1;

run.bcMode.x = ENUM.BCMODE_STATIC;

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % just to pop it into cylindrical geometry mode
%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

if grid(3) == 1
    run.activeSlices.xy  = true;
else
    run.activeSlices.xyz = true;
end
run.ppSave.dim3 = 10;

%run.VTOSettings = [1 1] / (31e6*10);
run.checkpointSteps = 100;

run.info        = 'Shearing dusty box simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 10;
  rp.firstCallIteration = 1;
  rp.spawnGUI = 1;
run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

%run.frameParameters.omega = 2.1715;
%run.frameParameters.rotateCenter = [256.5 256.5 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

