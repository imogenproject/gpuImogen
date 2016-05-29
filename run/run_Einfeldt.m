% Run Einfeldt Strong Rarefaction test.

%--- Initialize test ---%
grid = [1024 1 1];
run             = RiemannProblemInitializer(grid);
run.timeMax     = 0.1;
run.iterMax     = round(5*run.timeMax*grid(1)); % This will give steps max ~ 1.2x required

run.cfl         = .4;

if 1; % use Mach-based formula
    c0 = sqrt(run.gamma);
    mach = 1;
    t0 = 1 / (run.gamma-1) + .5 * run.gamma * mach^2;

    run.setupEinfeldt(mach, run.gamma)
else % use Einfelt (rho, m, n, e) formulation
%    run.rhol        = 1;
%    run.ml          = -1;
%    run.nl          = 0;
%    run.el          = 5;

%    run.rhor        = 1;
%    run.mr          = 1;
%    run.nr          = 0;
%    run.er          = 5;
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
  fm.iniMethod = 2; % hllc
  fm.toMethod = 2; % hllc. hll = 1
  fm.atstep = -1;
run.peripherals{end+1} = fm;
rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 0;

  rp.iterationsPerCall = 25;
  rp.firstCallIteration = 1;
run.peripherals{end+1} = rp;


%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

