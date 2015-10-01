
% Run sonic advection test at 3 different resolutions,
% Check accuracy & scaling

TestResults.name = 'Imogen Master Test Suite';

%--- Override: Run ALL the tests! ---%
doALLTheTests = 1;

%--- Individual selects ---%
% Advection/transport tests
doSonicAdvectStaticBG = 0;
doSonicAdvectMovingBG = 0;
doSonicAdvectAngleXY  = 0;
doEntropyAdvect       = 0;

% 1D tests
doSodTubeTests        = 0;
doEinfeldtTests       = 0;

% 2D tests
doCentrifugeTests     = 0;

% 3D tests
doSedovTests          = 1;

% As regards the choices of arbitrary inputs like Machs...
% If it works for $RANDOM_NUMBER_WITH_NO_PARTICULAR_SIGNIFICANCE
% It's probably right

% Picks how far we take the scaling tests
advectionDoublings  = 5;
einfeldtDoublings   = 9;
sodDoublings        = 9;
centrifugeDoublings = 6;

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG || doALLTheTests
    disp('Testing advection against stationary background.');
    try
        x = tsAdvection('sonic',[128 2 1], [1 0 0], [0 0 0], 0, advectionDoublings);
    catch ME
        fprintf('Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        prettyprintException(ME);
        x = 'FAILED';
    end
    TestResult.advection.Xalign_mach0 = x;
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG || doALLTheTests
    disp('Testing advection against moving background');
    try
        x = tsAdvection('sonic',[32 2 1], [1 0 0], [0 0 0], -.526172, advectionDoublings);
    catch ME
        fprintf('Oh dear: 1D Advection test simulation barfed.\nIf this wasn''t a mere syntax error, something is seriously wrong\nRecommend re-run full unit tests. Storing blank.\n');
        prettyprintExecption(ME);
        x = 'FAILED';
    end
        
    TestResult.advection.Xalign_mach0p5 = x;
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY || doALLTheTests
    disp('Testing advection in 2D across moving background');
    try
        x = tsAdvection('sonic',[64 64 1], [7 5 0], [0 0 0], .4387, advectionDoublings);
    catch ME
        fprintf('2D Advection test simulation barfed.\n');
        prettyprintException(ME);
        x = 'FAILED';
    end

    TestResult.advection.XY = x;
end

% This one is unhappy. It expects to have a variable defined (the self-rest-frame oscillation frequency) that isn't set for this type
%--- Test that an entropy wave just passively coasts along as it ought ---% 
%if doEntropyAdvect || doALLTheTests
%    TestResult.advection.HDentropy = tsAdvection('entropy',[1024 1024 1], [9 5 0], [0 0 0], 2.1948, doublings);
%end

%--- Run an Einfeldt double rarefaction test at the critical parameter ---%
if doEinfeldtTests || doALLTheTests
    disp('Testing convergence of Einfeldt tube');
    try
        x = tsEinfeldt(32, 1.4, 5, einfeldtDoublings);
    catch ME
        fprintf('Einfeldt tube test has failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
    end
    TestResult.einfeldt = x;
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests || doALLTheTests
    disp('Testing convergence of Sod tube');
    try
        x = tsSod(32,1,sodDoublings);
    catch ME
        fprintf('Sod shock tube test has failed.\n');
        prettyprintException(ME);
        x = 'FAILED';
    end
    
    TestResult.sod.X = x
end

if doCentrifugeTests || doALLTheTests
    disp('Testing centrifuge equilibrium-maintainence.');
    % Test centrifuge starting from 32x32 up to 1024x1024 with mildly
    % supersonic conditions
    try
        x = tsCentrifuge([32 32 1], 1.5, centrifugeDoublings); 
    catch ME
        disp('Centrifuge test has failed.');
        prettyprintException(ME);
        x = 'FAILED';
    end

    TestResult.centrifuge = x;
    % FIXME: Run a centrifuge with a rotating frame term!
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
    disp('Testing 2D Sedov-Taylor explosion');
    try
        x = tsSedov([32 32 1], [1 2 4]);
    catch ME
        disp('2D Sedov-Taylor test has failed.');
        prettyprintException(ME);
        x = 'FAILED';
    end
    TestResult.sedov2d = x;

    disp('Testing 3D Sedov-Taylor explosion');
    try
        x = tsSedov([32 32 32], [1 2]);
    catch ME
        disp('3D Sedov-Taylor test has failed.');
        prettyprintException(ME);
        x = 'FAILED';
    end

    TestResult.sedov3d = x;
end

%%%%%
% Test standing shocks, HD and MHD, 1/2/3 dimensional



%%% SOURCE TESTS???
% Test constant gravity field, (watch compressible water slosh?)

% Test RT magnetically-stabilized transverse oscillation?
% REQUIRES MAGNETISM

% Test behavior of radiative flow
% REQUIRES MAGNETISM

% Test rotating frame
% USE CENTRIFUGE TEST

if mpi_amirank0()
    save('~/FullTestSuiteResults.mat','TestResult');
end


