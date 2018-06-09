% Run Einfeldt Strong Rarefaction test.
M0 = 4;
%--- Initialize test ---%
grid = [512 1 1];
run             = RiemannProblemInitializer(grid);
run.timeMax     = 0.1; % about .08 is better for high Mach (M > 10) conditions
run.timeMax = .92*.5/((1+M0)*sqrt(run.gamma));
run.iterMax     = round(50*run.timeMax*grid(1)); % This will give steps max ~ 1.2x required

run.bcMode.x    = ENUM.BCMODE_CONSTANT;

if 1
    % uses a Mach-based formula
    % The left (right) halfslabs have speeds of - (+) mach(1) with soundspeed normalized to sqrt(gamma)
    % if mach(2) exists, vy values are set to +- that in the same fashion
    c0 = sqrt(run.gamma);
    mach = [M0 0];

    run.setupEinfeldt(mach, run.gamma)
else % use Einfelt's (rho, m, n, e) formulation (See Einfeldt 1988)
    run.setupEinfeldt([1 -1 0 5], [1 1 0 5]);
end

% Note that with HLL, a Delta m of greater than 2.1 results in too great of a rarefaction, resulting in negative mass density and NAN-Plague.
% With HLLC, this value is between 2.6 and 2.8
run.alias       = 'ET';
run.info        = 'Einfeldt Strong Rarefaction test.';
run.notes        = '';
run.ppSave.dim2 = 50;

% TEST DEMO
% Instructs the simulation to start with HLLC, then hop to HLL after 20 steps
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL; % hll
  fm.toMethod = ENUM.CFD_HLL; % hll. hll = 1
  fm.atstep = -1;
run.peripherals{end+1} = fm;
rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.spawnGUI = 1;
  rp.forceRedraw = 1;
  rp.iterationsPerCall = 1;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;


%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

