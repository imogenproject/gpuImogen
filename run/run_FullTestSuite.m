% Run Advection test.

%-- Initialize Imogen directory ---%
starterRun();

% Run sonic advection test at 3 different resolutions,
% Check raw accuracy & scaling

TestResults.name = 'Imogen Master Test Suite';

grid = [1024 8 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Override: Run ALL the tests! ---%
doFullBroadside = 0;

%--- Individual selects ---%
doSonicAdvectStaticBG = 0;
doSonicAdvectMovingBG = 0;
doSonicAdvectAngleXY  = 0;
doEntropyAdvect       = 0;
doSodTubeTests        = 1;

% As regards the choices of arbitrary inputs like Machs...
% If it works for $RANDOM_NUMBER_WITH_NO_PARTICULAR_SYMMETRIES_OR_SPECIALNESS
% It's probably right

%--- Gentle one-dimensional test: Advect a sound wave in X direction ---%
if doSonicAdvectStaticBG || doFullBroadside
    TestResults.advection.Xalign_mach0 = testAdvection('sonic',[1024 8 1], [8 0 0], 0);
end

%--- Test advection of a sound wave with the background translating at half the speed of sound ---%
if doSonicAdvectMovingBG || doFullBroadside
    TestResult.advection.Xalign_mach0p5 = testAdvection('sonic',[1024 8 1], [8 0 0], .526172);
end

%--- Test a sound wave propagating in a non grid aligned direction at supersonic speed---%
if doSonicAdvectAngleXY || doFullBroadside
    TestResult.advection.XY = testAdvection('sonic',[1024 1024 1], [7 5 0], .4387);
end

%--- Test that an entropy wave just passively coasts along as it ought ---% 
if doEntropyAdvect || doFullBroadside
    TestResult.advection.HDentropy = testAdvection('entropy',[1024 1024 1], [9 5 0], [0 0 0], 2.1948);
end

%--- Test the Sod shock tube for basic shock-capturingness ---%
if doSodTubeTests || doFullBroadside
    TestResult.sod.X = testSod(512,'X');
end


%--- Test propagation across a three-dimensional grid ---%
% Not sure that this mode actually works in the analyzer, though it will certainly work in the initializer
%if doSonicAdvectXYZ
%    TestResult.advection.XYZ = test
%end
%--- Run same battery of tests with entropy wave, magnetized entropy wave, MHD waves ---%

% Run OTV at moderate resolution, compare difference blah blah blah


%%% 3D tests

% Run Sedov-Taylor explosion at 2 resolutions,
% check results


%%%%%
% Test standing shocks, HD and MHD, 1/2/3 dimensional



%%% SOURCE TESTS
% Test constant gravity field, (watch compressible water slosh?)


% Test RT "instability" in magnetically stabilized fluid

% Test behavior of radiative flow

% Test rotating frame

% Try to run an accretion analysis?


