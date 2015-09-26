
% Run sonic advection test at 3 different resolutions,
% Check accuracy & scaling

TestResults.name = 'Imogen Master Test Suite';

%--- Override: Run ALL the tests! ---%
doALLTheTests = 0;

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
% If it works for $RANDOM_NUMBER_WITH_NO_SYMMETRY_OR_SPECIALNESS
% It's probably right

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG || doALLTheTests
    TestResult.advection.Xalign_mach0 = tsAdvection('sonic',[128 8 1], [1 0 0], [0 0 0], 0);
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG || doALLTheTests
    TestResult.advection.Xalign_mach0p5 = tsAdvection('sonic',[32 2 1], [1 0 0], [0 0 0], -.526172);
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY || doALLTheTests
    TestResult.advection.XY = tsAdvection('sonic',[1024 1024 1], [7 5 0], [0 0 0], .4387);
end

%--- Test that an entropy wave just passively coasts along as it ought ---% 
if doEntropyAdvect || doALLTheTests
    TestResult.advection.HDentropy = tsAdvection('entropy',[1024 1024 1], [9 5 0], [0 0 0], 2.1948);
end

%--- Run an Einfeldt double rarefaction test at the critical parameter ---%
if doEinfeldtTests || doALLTheTests
    TestResult.einfeldt = tsEinfeldt(32, 5, 1.4, 5);
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests || doALLTheTests
    TestResult.sod.X = tsSod(32,1,6);
end

if doCentrifugeTests || doALLTheTests
    % Test centrifuge starting from 32x32 up to 1024x1024 with mildly
    % supersonic conditions
    TestResult.centrifuge = tsCentrifuge([32 32 1], 6, 1.5); 
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
	TestResult.sedov3d = tsSedov([32 32 32], [1 2 3 4]);
	TestResult.sedov2d = tsSedov([32 32 1], [1 2 4 8 16]);
end


%%%%%
% Test standing shocks, HD and MHD, 1/2/3 dimensional



%%% SOURCE TESTS???
% Test constant gravity field, (watch compressible water slosh?)

% Test RT magnetically-stabilized transverse oscillation?

% Test behavior of radiative flow

% Test rotating frame

% Try to run an accretion analysis?

save('~/FullTestSuiteResults.mat','TestResult');

