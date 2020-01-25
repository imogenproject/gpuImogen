% Run sonic advection test at 3 different resolutions,
% Check accuracy & scalingre
format long

TestResults.name = 'Imogen Master Test Suite';

TestResultFilename = '~/mastertest_1d_serial';

realtimePictures = 0;

%--- Overrides: Run ALL the tests! ---%
doALLTheTests = 0;
doAll1DTests  = 0;
doAll2DTests  = 0;
doAll3DTests  = 0;

%--- Individual selects ---%
% Advection/transport tests
% 1D
doSonicAdvectStaticBG   = 0;
doSonicAdvectMovingBG   = 0;
doEntropyAdvectStaticBG = 0;
doEntropyAdvectMovingBG = 0;
doDustyWaveStaticBG     = 0;
doDustyWaveMovingBG     = 0;
% 2D
doSonicAdvectAngleXY    = 0;

% 1D tests
doSodTubeTests        = 0;
doEinfeldtTests       = 0;
doDoubleBlastTests    = 0;
doNohTubeTests        = 0;
doDustyBoxes          = 0;

% 2D tests
doCentrifugeTests     = 0;
doSedov2DTests        = 0;

% 3D tests
doSedov3DTests        = 1;

% Apply the "do ALL the things" overrides to the individual run/dontrun variables above
% without cluttering this file up
maskSuiteTestsOn

% As regards the choices of arbitrary inputs like Machs...
% If it works for $RANDOM_NUMBER_WITH_NO_PARTICULAR_SIGNIFICANCE
% It's probably right

baseResolution = 16;

SaveManager.logPrint('NOTICE: Base resolution is currently set to %i.\nNOTICE: If the number of MPI ranks or GPUs will divide this to below 6, things will Break.\n', baseResolution);

% Picks how far we take the scaling tests in 1D
ftn = 5;
advectionDoublings  = ftn;
doubleblastDoublings= ftn;
dustyBoxDoublings   = ftn;
einfeldtDoublings   = ftn;
nohDoublings        = ftn;
sodDoublings        = ftn;

% Choose number of refinements in 2D
advection2DDoublings= 3;
centrifugeDoublings = 3;
sedov2D_scales      = [1 2 4];

% And in 3D
sedovBase = 24;
sedov3D_scales      = [1 2 3];

fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLL;
  %fm.toMethod = ENUM.CFD_HLLC;
  %fm.atstep = 3;

exceptionList = {};

startSimulationsTime = clock();

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG
    SaveManager.logPrint('Testing advection against stationary background.\n');
    try
        x = tsAdvection('sonic',[baseResolution 1 1], [1 0 0], [0 0 0], 0, advectionDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.Xalign_mach0 = x;
    if mpi_amirank0(); disp('Results for advection against stationary background:'); disp(x); end
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG
    SaveManager.logPrint('Testing advection against moving background\n');
    try
        x = tsAdvection('sonic',[baseResolution 1 1], [1 0 0], [0 0 0], -.526172, advectionDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
        
    TestResult.advection.Xalign_mach0p5 = x;
    if mpi_amirank0(); disp('Results for advection against moving background:'); disp(x); end
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY
    SaveManager.logPrint('Testing sound advection''s prefactor in 2D\n');
    try
        x = tsCrossAdvect('sonic', [256 256 1], [2 2 0], [0 0 0], [0 0 0], realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, '2D Advection test simulation barfed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end

    TestResult.advection.XY = x;
    if mpi_amirank0(); disp('Results for cross-grid sonic advection with moving background:'); disp(x); end
end

% This one is unhappy. It expects to have a variable defined (the self-rest-frame oscillation frequency) that isn't set for this type
%--- Test that an entropy wave just passively coasts along as it ought ---% 
if doEntropyAdvectStaticBG
    SaveManager.logPrint('Testing entropy wave advection in 1D with static background.\n');
    try
        x = tsAdvection('entropy',[baseResolution 1 1], [1 0 0], [0 0 0], 0, advectionDoublings);
    catch ME
        prettyprintException(ME, 0, 'Stationary entropy wave test simulation failed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.HDentropy_static = x;
end

if doEntropyAdvectMovingBG || doALLTheTests
    SaveManager.logPrint('Testing convected entropy wave in 1D.\n');
    try
        x = tsAdvection('entropy',[baseResolution 1 1], [1 0 0], [0 0 0], 1.178, advectionDoublings);
    catch ME
        prettyprintException(ME, 0, 'Convected entropy wave test simulation failed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.HDentropy_moving = x;
end

%--- Test the dusty wave solution ---%
if doDustyWaveStaticBG
    SaveManager.logPrint('Testing dustywave with static BG.\n');
    try
        x = tsDustywave([baseResolution 1 1], advectionDoublings, .1, realtimePictures, fm);
        y = tsDustywave([baseResolution 1 1], advectionDoublings, 1, realtimePictures, fm);
        z = tsDustywave([baseResolution 1 1], advectionDoublings, 25, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Dustywave test failed.\n');
        x = 'FAILED'; y = 'FAILED'; z = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.advection.dustywave_k0p1 = x;
    TestResult.advection.dustywave_k1 = y;
    TestResult.advection.dustywave_k25 = z;
end

%--- Run an Einfeldt double rarefaction test at the critical parameter ---%
if doEinfeldtTests
    SaveManager.logPrint('Testing convergence of Einfeldt tube.\n');
    try
        x = tsEinfeldt(baseResolution, 1.4, 4, einfeldtDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Einfeldt tube test has failed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    TestResult.einfeldt = x;
    if mpi_amirank0(); disp('Results for Einfeldt tube refinement:'); disp(x); end
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests
    SaveManager.logPrint('Testing convergence of Sod tube.\n');
    try
        x = tsSod(baseResolution, 1, sodDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Sod shock tube test has failed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    
    TestResult.sod.X = x;
    if mpi_amirank0(); disp('Results for Sod tube refinement:'); disp(x); end
end

if doNohTubeTests
    SaveManager.logPrint('Testing convergence of Noh tube.\n');
    try
        x = tsNohtube(baseResolution, nohDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Noh shock tube test has failed.\n');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end
    
    TestResult.noh.X = x;
    if mpi_amirank0(); disp('Results for Noh tube refinement:'); disp(x); end
end

if doCentrifugeTests
    SaveManager.logPrint('Testing centrifuge equilibrium-maintainence.\n');
    try
        x = tsCentrifuge([baseResolution baseResolution 1], 1.5, centrifugeDoublings, realtimePictures, fm);
    catch ME
        prettyprintException(ME, 0, 'Centrifuge test has failed.');
        x = 'FAILED';
        exceptionList{end+1} = ME;
    end

    TestResult.centrifuge = x;
    if mpi_amirank0(); disp('Results for centrifuge test in stationary frame:'); disp(x); end
    % FIXME: Run a centrifuge with a rotating frame term!
end

if doDoubleBlastTests
   SaveManager.logPrint('Performing refinement test on WC1984 double-blast.\n');
   try
       x = tsDoubleBlast([baseResolution 1 1], doubleblastDoublings, realtimePictures, fm);
   catch ME
        prettyprintException(ME, 0, 'Double blast test has failed.');
        x = 'FAILED';
        exceptionList{end+1} = ME;
   end
   
   TestResult.doubleBlast = x;
   if mpi_amirank0(); disp('Results for double blast test:'); disp(x); end
   
end

if doDustyBoxes
   SaveManager.logPrint('Performing temporal refinement test on spatially uniform dusty boxes.\n');
   x1 = 'FAILED'; x2 = 'FAILED'; x3 = 'FAILED';
   try
       x1 = tsDustybox(baseResolution, 5/3, .01, ENUM.MULTIFLUID_ETDRK1, dustyBoxDoublings, realtimePictures, fm);
       x2 = tsDustybox(baseResolution, 5/3, .01, ENUM.MULTIFLUID_LOGTRAP2, dustyBoxDoublings, realtimePictures, fm);
       x3 = tsDustybox(baseResolution, 5/3, .01, ENUM.MULTIFLUID_LOGTRAP3, dustyBoxDoublings, realtimePictures, fm);
   catch ME
        prettyprintException(ME, 0, 'Dusty box test (Mach .01) has failed.');
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.slow_etd = x1;
   TestResult.dustybox.slow_lt2 = x2;
   TestResult.dustybox.slow_lt3 = x3;
   if mpi_amirank0()
       disp('Results for M=.01 dusty box test using ETDRK1:'); disp(x1);
       disp('Results for M=.01 dusty box test using LogTrap2:'); disp(x2);
       disp('Results for M=.01 dusty box test using LogTrap3:'); disp(x3);
   end

   x1 = 'FAILED'; x2 = 'FAILED'; x3 = 'FAILED';
   try
       x1 = tsDustybox(baseResolution, 5/3, .20, ENUM.MULTIFLUID_ETDRK1, dustyBoxDoublings, realtimePictures, fm);
       x2 = tsDustybox(baseResolution, 5/3, .20, ENUM.MULTIFLUID_LOGTRAP2, dustyBoxDoublings, realtimePictures, fm);
       x3 = tsDustybox(baseResolution, 5/3, .20, ENUM.MULTIFLUID_LOGTRAP3, dustyBoxDoublings, realtimePictures, fm);
   catch ME
        prettyprintException(ME, 0, 'Dusty box test (Mach 0.20) has failed.');
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.mid_etd = x1;
   TestResult.dustybox.mid_lt2 = x2;
   TestResult.dustybox.mid_lt3 = x3;
   if mpi_amirank0()
       disp('Results for M=.01 dusty box test using ETDRK1:'); disp(x1);
       disp('Results for M=.01 dusty box test using LogTrap2:'); disp(x2);
       disp('Results for M=.01 dusty box test using LogTrap3:'); disp(x3);
   end

   x1 = 'FAILED'; x2 = 'FAILED'; x3 = 'FAILED';
   try
       x1 = tsDustybox(baseResolution, 5/3, 2.0, ENUM.MULTIFLUID_ETDRK1, dustyBoxDoublings, realtimePictures, fm);
       x2 = tsDustybox(baseResolution, 5/3, 2.0, ENUM.MULTIFLUID_LOGTRAP2, dustyBoxDoublings, realtimePictures, fm);
       x3 = tsDustybox(baseResolution, 5/3, 2.0, ENUM.MULTIFLUID_LOGTRAP3, dustyBoxDoublings, realtimePictures, fm);
   catch ME
        prettyprintException(ME, 0, 'Dusty box test (Mach 2) has failed.');
        exceptionList{end+1} = ME;
   end
   TestResult.dustybox.supersonic_etd = x1;
   TestResult.dustybox.supersonic_lt2 = x2;
   TestResult.dustybox.supersonic_lt3 = x3;
   if mpi_amirank0()
       disp('Results for M=.01 dusty box test using ETDRK1:'); disp(x1);
       disp('Results for M=.01 dusty box test using LogTrap2:'); disp(x2);
       disp('Results for M=.01 dusty box test using LogTrap3:'); disp(x3);
   end

end

%--- Test propagation across a three-dimensional grid ---%
% Not sure that this mode actually works in the analyzer, though it will certainly work in the initializer
%if doSonicAdvectXYZ
%    TestResult.advection.XYZ = test
%end
%--- Run same battery of tests with entropy wave, magnetized entropy wave, MHD waves ---%

% Run OTV at moderate resolution, compare difference blah blah blah

%%% 3D tests
if doSedov2DTests
    if ~isempty(sedov2D_scales)
    SaveManager.logPrint('Testing 2D Sedov-Taylor explosion.\n');
        try
            x = tsSedov([sedovBase sedovBase 1], sedov2D_scales, realtimePictures, fm);
        catch ME
            prettyprintException(ME, 0, '2D Sedov-Taylor test has failed.');
            x = 'FAILED';
            exceptionList{end+1} = ME;
        end
        TestResult.sedov2d = x;
    if mpi_amirank0(); disp('Results for 2D (cylindrical) Sedov-Taylor explosion:'); disp(x); end
    end
end

if doSedov3DTests
    if ~isempty(sedov3D_scales)
        SaveManager.logPrint('Testing 3D Sedov-Taylor explosion.\n');
        try
            x = tsSedov([sedovBase sedovBase sedovBase], sedov3D_scales, realtimePictures, fm);
        catch ME
            prettyprintException(ME, 0, '3D Sedov-Taylor test has failed.');
            x = 'FAILED';
            exceptionList{end+1} = ME;
        end
        TestResult.sedov3d = x;
        if mpi_amirank0(); disp('Results for 3D (spherical) Sedov-Taylor explosion:'); disp(x); end
    end

end

endSimulationsTime = clock();

%%%%%
% Test standing shocks, HD and MHD, 1/2/3 dimensional?

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
    disp(['Total runtime of all simulations measured by rank 0 was ' num2str(t) ' sec.']);
    disp(['Simulation output is stored in file ' TestResultFilename '.mat'])
    if numel(exceptionList) > 0
        disp(['A total of ' num2str(numel(exceptionList)) ' errors were encountered']);
        for k = 1:numel(exceptionList)
            disp(['====================  Error number ' num2str(k) ':']);
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





