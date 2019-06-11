
grid = [1024 256 1];
%grid = [1080 900 384];

run                 = ShearingBoxInitializer(grid);
run.iterMax         = 100000;

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
run.azimuthalMode = 50;

% Numeric parameters:
% - densityCutoffFraction: stop density falloff below rho = f*rho_max
% - useZMirror: simulate only upper half of disk, Z- BC automatically set to mirror
% - normalizationRadius - r0
% - normalizeValues - 

run.normalizationRadius = 150e10; % 10AU
run.normalizeValues = 1;

% 5 to 15 AU
run.innerRadius = 150e9*7.5; % 5AU
run.outerRadius = 150e9*12.5;% 40AU

run.Sigma0 = 1000; % 100g/cm^2 = 1000kg/m^2
run.cs0 = run.cs0; % low temp
run.densityExponent = -1; % steady state accretion disk
run.temperatureExponent = -0.66; % P_solar ~ r^-2 = P_bbody ~ T^4 -> T ~ r^-0.5

%run.solarWindAngle = 45;
run.solarWindRhoRef = .75e-7;
%run.solarWindTref = 250; % kelvin
%run.solarWindEscapeFactor = 1.5;

run.dustFraction = .21; % no dust at all this time!
run.hack_dscale = 200;
run.hack_dustvrad = -.005;

run.gasPerturb = .030; % disturb at 5% soundspeed

run.useZMirror = 0;

run.bcMode.x = {ENUM.BCMODE_STATIC, ENUM.BCMODE_FREEBALANCE};
run.bcMode.z = ENUM.BCMODE_FREEBALANCE; % test hack

run.geomgr.geometryCylindrical(1, 1, 1, 0, 1); % just to pop it into cylindrical geometry mode
%run.image.interval  = 50;
%run.image.speed     = true;
%run.image.mass      = true;

if grid(3) == 2
    run.activeSlices.xy  = true;
else
    run.activeSlices.xyz = true;
end
run.ppSave.dim3 = 10;

%run.VTOSettings = 200*[1 1] / (2*pi);
%run.checkpointSteps = -1;

run.compositeSourceOrders = [4 4];

run.info        = 'Shearing dusty box simulation';
run.notes       = '';

rp = RealtimePlotter();
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 10;
  rp.firstCallIteration = 1;
  rp.spawnGUI = 1;
  
  rp.plotmode = 2;
rp.cut = [128 1 1];
rp.indSubs = [8 1 256;1 1 256;1 1 1];
rp.movieProps(1, 0, 'RTP_');
rp.vectorToPlotprops(1, [1  11   0   4   1   0   1   0   1   1  12   1   8   1   0   0   0]);
rp.vectorToPlotprops(2, [2   5   0   4   1   0   1   0   0   1  10   1   8   1   0   0   0]);

run.peripherals{end+1} = rp;

fm = FlipMethod();
fm.iniMethod = ENUM.CFD_HLL;
run.peripherals{end+1} = fm;

run.image.parallelUniformColors = true;

%run.frameParameters.omega = 2.1715;
%run.frameParameters.rotateCenter = [256.5 256.5 0];

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToStructure();
    imogen(icfile);
end

