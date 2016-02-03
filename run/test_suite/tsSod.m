function result = tsSod(N, direct, doublings, prettyPictures)

if nargin < 4
    prettyPictures = 0;
end

%--- Initialize test ---%
run         = SodShockTubeInitializer([N 2 1]);
run.normal(direct);
run.iterMax     = 50000;
run.timeMax     = 0.25;

run.alias       = '';
run.info    = 'Sod shock tube test.';
run.notes       = 'Simple axis aligned shock tube test';

run.ppSave.dim2 = 100;

run.bcMode.x = ENUM.BCMODE_CONST;

if prettyPictures
    run.useInSituAnalysis = 1;
    run.stepsPerInSitu = 25;
    run.inSituHandle = @RealtimePlotter;
    instruct.plotmode = 1;

    instruct.plotDifference = 0;
    instruct.pause = 0;

    run.inSituInstructions = instruct;
end

%--- Run tests ---%

result.L1    = [];
result.L2    = [];
result.res   = [];
result.paths = {};

for p = 1:doublings;
    % Run test at given resolution
    run.grid(direct) = N*2^(p-1);
    icfile           = run.saveInitialCondsToFile();
    outpath          = imogen(icfile);
    enforceConsistentView(outpath);

    % Load last frame
    S = SavefilePortal(outpath);
    S.setFrametype(7);
    u = S.jumpToLastFrame();

    % Compute L_n integral error norms and output
    T = sum(u.time.history);
    X = SodShockSolution(run.grid(direct), T);

    result.L2(p)    = sqrt(mpi_sum(norm(u.mass(:,1)-X.mass',2).^2) / mpi_sum(numel(X.mass)) );
    result.L1(p)    = mpi_sum(norm(u.mass(:,1)-X.mass',1)) / mpi_sum(numel(X.mass));
    result.res(p)   = run.grid(direct);
    result.paths{p} = outpath;

end

% Compute convergence order
dlogh = -log(2);

result.OrderL1 = diff(log(result.L1)) / dlogh;
result.OrderL2 = diff(log(result.L2)) / dlogh;

end
