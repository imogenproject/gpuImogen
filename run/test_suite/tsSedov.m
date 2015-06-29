function result = tsSedov(iniResolution, multiples)

grid = iniResolution;

%--- Initialize test ---%
% Default ST blast desposits E=1 into a region 3.5 cells in radius
run         = SedovTaylorBlastWaveInitializer(grid);

run.autoEndtime = 1; % Automatically run until Rblast = 0.45

run.alias   = 'SEDOV_ts';
run.info    = 'Sedov-Taylor blast wave convergence test.';
run.notes   = 'Eblast=1, box diameter = [1 1 1], Rend = 0.45';

result.paths = {};
result.times = [];
result.rhoL1 = [];
result.rhoL2 = [];

%--- Run tests ---%
for N = 1:numel(multiples)
    grid = iniResolution*multiples(N);
    if iniResolution(3) == 1; grid(3) = 1; end % keep 2D, 2D

    run.grid = grid;
    icfile = run.saveInitialCondsToFile();
    outdir = imogen(icfile);
    status = analyzeSedovTaylor(outdir);

    result.paths{end+1} = outdir;

    % Take times from the first run
    if N == 1; result.times = status.time; end
    
    result.rhoL1(end+1,:) = status.rhoL1;
    result.rhoL2(end+1,:) = status.rhoL2;
end

end

