
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
doEinfeldtTests       = 1;

% 2D tests
doCentrifugeTests     = 0;

% 3D tests
doSedovTests          = 0 ;

% As regards the choices of arbitrary inputs like Machs...
% If it works for $RANDOM_NUMBER_WITH_NO_PARTICULAR_SIGNIFICANCE
% It's probably right

% Picks how far we take the scaling tests
advectionDoublings  = 6;
einfeldtDoublings   = 9;
sodDoublings        = 9;
centrifugeDoublings = 6;

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG || doALLTheTests
    TestResult.advection.Xalign_mach0 = tsAdvection('sonic',[128 2 1], [1 0 0], [0 0 0], 0, advectionDoublings);
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG || doALLTheTests
    TestResult.advection.Xalign_mach0p5 = tsAdvection('sonic',[32 2 1], [1 0 0], [0 0 0], -.526172, advectionDoublings);
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY || doALLTheTests
    TestResult.advection.XY = tsAdvection('sonic',[64 64 1], [7 5 0], [0 0 0], .4387, advectionDoublings);
end

% This one is unhappy. It expects to have a variable defined (the self-rest-frame oscillation frequency) that isn't set for this type
%--- Test that an entropy wave just passively coasts along as it ought ---% 
%if doEntropyAdvect || doALLTheTests
%    TestResult.advection.HDentropy = tsAdvection('entropy',[1024 1024 1], [9 5 0], [0 0 0], 2.1948, doublings);
%end

%--- Run an Einfeldt double rarefaction test at the critical parameter ---%
if doEinfeldtTests || doALLTheTests
    TestResult.einfeldt = tsEinfeldt(32, 1.4, 5, einfeldtDoublings);
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests || doALLTheTests
    TestResult.sod.X = tsSod(32,1,sodDoublings);
end

if doCentrifugeTests || doALLTheTests
    % Test centrifuge starting from 32x32 up to 1024x1024 with mildly
    % supersonic conditions
    TestResult.centrifuge = tsCentrifuge([32 32 1], 1.5, centrifugeDoublings); 

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
	TestResult.sedov3d = tsSedov([32 32 32], [1 2 4 8]);
	TestResult.sedov2d = tsSedov([32 32 1], [1 2 4 8 16 32]);
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


