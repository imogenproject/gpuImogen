%   Run Rayleigh-Taylor instability shock test.

%--- Initialize test ---%
grid = [512 512 1];
run         = RayleighTaylorInitializer(grid);
run.iterMax = 500;

run.rhoTop           = 2;
run.rhoBottom        = 1;
run.P0               = 2.5;
run.gravity.constant = .1;
run.bcMode.y = {ENUM.BCMODE_MIRROR,ENUM.BCMODE_STATIC};


run.pertAmplitude = .2;
run.randomPert    = 1;
run.Kz            = 0;
%run.Kx            = 4;

%run.ppSave.dim2 = 10;
run.image.interval = 25;
run.image.mass     = true;
run.info           = 'Rayleigh Taylor instability test';
run.notes          = '';

%run.B0               = 0;
%run.bcMode.gravity.y = 'const';
%run.image.pGas       = true;
%run.image.speed      = true;

        run.useInSituAnalysis = 1;
        run.stepsPerInSitu = 20;
        run.inSituHandle = @RealtimePlotter;
                instruct.plotmode = 4;
                instruct.plotDifference = 0;
                instruct.pause = 0;
        run.inSituInstructions = instruct;

%--- Run tests ---%
if (true) %Primary test
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

