% Run sonic advection test at 3 different resolutions,
% Check accuracy & scaling
format long

TestResults.name = 'Imogen Master Test Suite';

TestResultFilename = '~/fulltestasdf';

%--- Override: Run ALL the tests! ---%
doALLTheTests = 0;
realtimePictures = 0;

%--- Individual selects ---%
% Advection/transport tests
doSonicAdvectStaticBG   = 0;
doSonicAdvectMovingBG   = 0;
doSonicAdvectAngleXY    = 0;
doEntropyAdvectStaticBG = 0;
doEntropyAdvectMovingBG = 0;
doDustyWaveStaticBG     = 0;
doDustyWaveMovingBG     = 0;

% 1D tests
doSodTubeTests        = 0;
doEinfeldtTests       = 1;
doDoubleBlastTests    = 0;
doNohTubeTests        = 0;
doDustyBoxes          = 0;

% 2D tests
doCentrifugeTests     = 0;

% 3D tests
doSedovTests          = 0;

% As regards the choices of arbitrary inputs like Machs...
% If it works for $RANDOM_NUMBER_WITH_NO_PARTICULAR_SIGNIFICANCE
% It's probably right

baseResolution = 32;

if mpi_amirank0()
    fprintf('NOTICE: Base resolution is currently set to %i.\nNOTICE: If the number of MPI ranks or GPUs will divide this to below 6, things will Break.\n', baseResolution);
end

% Picks how far we take the scaling tests
advectionDoublings  = 5;
doubleblastDoublings= 6;
dustyBoxDoublings   = 7;
einfeldtDoublings   = 4;
nohDoublings        = 7;
sodDoublings        = 7;

advection2DDoublings= 6;
centrifugeDoublings = 6;
sedov2D_scales      = [1 2 4 8 16 24];

sedov3D_scales      = 1;


fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL;
  %fm.toMethod = ENUM.CFD_HLLC;
  %fm.atstep = 3;

exceptionList = {};

startSimulationsTime = clock();

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG || doALLTheTests
    if mpi_amirank0(); disp('Testing advection against stationary background.'); end
    try
        x = tsAdvection('sonic',[baseResolution 1 1], [1 0 0], [0 0 0], 0, advectionDoublings, realtimePictures, fm);
    catch ME
        fprintf('Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.Xalign_mach0 = x;
    if mpi_amirank0(); disp('Results for advection against stationary background:'); disp(x); end
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG || doALLTheTests
    if mpi_amirank0(); disp('Testing advection against moving background'); end
    try
        x = tsAdvection('sonic',[baseResolution 1 1], [1 0 0], [0 0 0], -.526172, advectionDoublings, realtimePictures, fm);
    catch ME
        fprintf('Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
        
    TestResult.advection.Xalign_mach0p5 = x;
    if mpi_amirank0(); disp('Results for advection against moving background:'); disp(x); end
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY || doALLTheTests
    if mpi_amirank0(); disp('Testing sound advection''s prefactor in 2D'); end
    try
        x = tsCrossAdvect('sonic', [256 256 1], [1 0 0], [0 0 0], [0 0 0], realtimePictures, fm);
    catch ME
        fprintf('2D Advection test simulation barfed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end

    TestResult.advection.XY = x;
    if mpi_amirank0(); disp('Results for cross-grid sonic advection with moving background:'); disp(x); end
end

% This one is unhappy. It expects to have a variable defined (the self-rest-frame oscillation frequency) that isn't set for this type
%--- Test that an entropy wave just passively coasts along as it ought ---% 
if doEntropyAdvectStaticBG || doALLTheTests
    if mpi_amirank0(); disp('Testing entropy wave advection in 1D with static background.'); end
    try
        TestResult.advection.HDentropy_static = tsAdvection('entropy',[baseResolution 1 1], [1 0 0], [0 0 0], 0, advectionDoublings);
    catch ME
        fprintf('Stationary entropy wave test simulation failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
end
if doEntropyAdvectMovingBG || doALLTheTests
    if mpi_amirank0(); disp('Testing convected entropy wave in 1D.'); end
    try
        TestResult.advection.HDentropy_moving = tsAdvection('entropy',[baseResolution 1 1], [1 0 0], [0 0 0], 1.178, advectionDoublings);
    catch ME
        fprintf('Convected entropy wave test simulation failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
end

%--- Test the dusty wave solution ---%
if doDustyWaveStaticBG || doALLTheTests
    if mpi_amirank0(); disp('Testing dustywave with static BG'); end
    try
        x = tsDustywave([baseResolution 1 1], advectionDoublings, .1, realtimePictures, fm);
        y = tsDustywave([baseResolution 1 1], advectionDoublings, 1, realtimePictures, fm);
        z = tsDustywave([baseResolution 1 1], advectionDoublings, 25, realtimePictures, fm);
    catch ME
        fprintf('Dustywave test failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.dustywave_k0p1 = x;
    TestResult.advection.dustywave_k1 = y;
    TestResult.advection.dustywave_k25 = z;
end

%--- Run an Einfeldt double rarefaction test at the critical parameter ---%
if doEinfeldtTests || doALLTheTests
    if mpi_amirank0(); disp('Testing convergence of Einfeldt tube'); end
    try
        x = tsEinfeldt(baseResolution, 1.4, 4, einfeldtDoublings, realtimePictures, fm);
    catch ME
        fprintf('Einfeldt tube test has failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.einfeldt = x;
    if mpi_amirank0(); disp('Results for Einfeldt tube refinement:'); disp(x); end
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests || doALLTheTests
    if mpi_amirank0(); disp('Testing convergence of Sod tube'); end
    try
        x = tsSod(baseResolution, 1, sodDoublings, realtimePictures, fm);
    catch ME
        fprintf('Sod shock tube test has failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    
    TestResult.sod.X = x;
    if mpi_amirank0(); disp('Results for Sod tube refinement:'); disp(x); end
end

if doNohTubeTests || doALLTheTests
    if mpi_amirank0(); disp('Testing convergence of Noh tube'); end
    try
        x = tsNohtube(baseResolution, nohDoublings, realtimePictures, fm);
    catch ME
        fprintf('Noh shock tube test has failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    
    TestResult.noh.X = x;
    if mpi_amirank0(); disp('Results for Noh tube refinement:'); disp(x); end
end

if doCentrifugeTests || doALLTheTests
    if mpi_amirank0(); disp('Testing centrifuge equilibrium-maintainence.'); end
    try
        x = tsCentrifuge([baseResolution baseResolution 1], 1.5, centrifugeDoublings, realtimePictures, fm);
    catch ME
        disp('Centrifuge test has failed.');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end

    TestResult.centrifuge = x;
    if mpi_amirank0(); disp('Results for centrifuge test in stationary frame:'); disp(x); end
    % FIXME: Run a centrifuge with a rotating frame term!
end

if doDoubleBlastTests || doALLTheTests
   if mpi_amirank0(); disp('Performing refinement test on WC1984 double-blast'); end
   try
       x = tsDoubleBlast([baseResolution 1 1], doubleblastDoublings, realtimePictures, fm);
   catch ME
       disp('Double blast test has failed.');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
   end
   
   TestResult.doubleBlast = x;
   if mpi_amirank0(); disp('Results for double blast test:'); disp(x); end
   
end

if doDustyBoxes || doALLTheTests
   if mpi_amirank0(); disp('Performing temporal refinement test on spatially uniform dusty boxes'); end
   try
       x = tsDustybox(baseResolution, 5/3, .01, dustyBoxDoublings, realtimePictures, fm);
   catch ME
       disp('Dusty box test (Mach .01) has failed.');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.slow = x;
   if mpi_amirank0(); disp('Results for M=.01 dusty box test:'); disp(x); end
   try
       x = tsDustybox(baseResolution, 5/3, .25, dustyBoxDoublings, realtimePictures, fm);
   catch ME
       disp('Dusty box test (Mach 0.25) has failed.');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.mid = x;
   if mpi_amirank0(); disp('Results for M=.25 dusty box test:'); disp(x); end
   try
       x = tsDustybox(baseResolution, 5/3, 2.0, dustyBoxDoublings, realtimePictures, fm);
   catch ME
       disp('Dusty box test (Mach 2) has failed.');
        prettyprintException(ME);
        x = 'FAILED';
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.supersonic = x;
   if mpi_amirank0(); disp('Results for M=2.0 dusty box test:'); disp(x); end
   
end

%--- Test propagation across a three-dimensional grid ---%
% Not sure that this mode actually works in the analyzer, though it will certainly work in the initializer
%if doSonicAdvectXYZ
%    TestResult.advection.XYZ = test
%end
%--- Run same battery of tests with entropy wave, magnetized entropy wave, MHD waves ---%

% Run OTV at moderate resolution, compare difference blah blah blah

%%% 3D tests
if doSedovTests || doALLTheTests
    if ~isempty(sedov2D_scales)
    if mpi_amirank0(); disp('Testing 2D Sedov-Taylor explosion'); end
        try
            x = tsSedov([baseResolution baseResolution 1], sedov2D_scales, realtimePictures, fm);
        catch ME
            disp('2D Sedov-Taylor test has failed.');
            prettyprintException(ME);
            x = 'FAILED';
            exceptionList{end+1} = ME;
        end
        TestResult.sedov2d = x;
    if mpi_amirank0(); disp('Results for 2D (cylindrical) Sedov-Taylor explosion:'); disp(x); end
    end

    if ~isempty(sedov3D_scales)
        if mpi_amirank0(); disp('Testing 3D Sedov-Taylor explosion'); end
        try
            x = tsSedov([baseResolution baseResolution baseResolution], sedov3D_scales, realtimePictures, fm);
        catch ME
            disp('3D Sedov-Taylor test has failed.');
            prettyprintException(ME);
            x = 'FAILED';
            exceptionList{end+1} = ME;
        end
        TestResult.sedov3d = x;
        if mpi_amirank0(); disp('Results for 3D (spherical) Sedov-Taylor explosion:'); disp(x); end
    end

end

endSimulationsTime = clock();

TestResult.einfeldt.L1
TestResult.einfeldt.L2

%%%%%
% Test standing shocks, HD and MHD, 1/2/3 dimensional

if mpi_amirank0()
    disp('========================================');

    if exist([TestResultFilename '.mat'], 'file') ~= 0
        disp(['!!! WARNING: filename "' TestResultFilename '.mat exists! Appending YYYYMMDD_HHMMSS.']);
        disp('========================================');
        TestResultFilename = [TestResultFilename date2ymdhms(startSimulationsTime)];
    end
    save([TestResultFilename '.mat'],'TestResult');

    disp('Full test suite is finished running.');
    t = etime(endSimulationsTime, startSimulationsTime);
    disp(['Total runtime of all simulations measured by rank 0 was' num2str(t) ' sec.']);
    disp(['Simulation output is stored in file ' TestResultFilename])
    if numel(exceptionList) > 0
        disp(['A total of ' num2str(numel(exceptionList)) ' errors were encountered']);
        for k = 1:numel(exceptionList)
            disp('====================');
            disp(['Error number ' num2str(k) ':']);
            prettyprintException(exceptionList{k});
        end
    else
        disp('No errors were reported by any simulations.');
    end
    disp('========================================');
end

%%% SOURCE TESTS???
% Test constant gravity field, (watch compressible water slosh?)

% Test RT magnetically-stabilized transverse oscillation?
% REQUIRES MAGNETISM

% Test behavior of radiative flow
% REQUIRES MAGNETISM

% Test rotating frame
% USE CENTRIFUGE TEST





