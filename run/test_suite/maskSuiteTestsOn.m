% This little bit of script-pasta hides the verbose process where doALLTheTests and doAll*DTests
% mask individual tests on from cluttering up the run_FullTestSuite script

doAll1DTests  = doAll1DTests | doALLTheTests;
doAll2DTests  = doAll2DTests | doALLTheTests;
doAll3DTests  = doAll3DTests | doALLTheTests;

%--- Individual selects ---%
% Advection/transport tests
% 1D
doSonicAdvectStaticBG   = doSonicAdvectStaticBG | doAll1DTests; 
doSonicAdvectMovingBG   = doSonicAdvectMovingBG | doAll1DTests;
doEntropyAdvectStaticBG = doEntropyAdvectStaticBG | doAll1DTests;
doEntropyAdvectMovingBG = doEntropyAdvectMovingBG | doAll1DTests;
doDustyWaveStaticBG     = doDustyWaveStaticBG   | doAll1DTests;
doDustyWaveMovingBG     = doDustyWaveMovingBG   | doAll1DTests;
% 2D
doSonicAdvectAngleXY    = doSonicAdvectAngleXY  | doAll2DTests;

% 1D tests
doSodTubeTests        = doSodTubeTests          | doAll1DTests;
doEinfeldtTests       = doEinfeldtTests         | doAll1DTests;
doDoubleBlastTests    = doDoubleBlastTests      | doAll1DTests;
doNohTubeTests        = doNohTubeTests          | doAll1DTests;
doDustyBoxes          = doDustyBoxes            | doAll1DTests;

% 2D tests
doCentrifugeTests     = doCentrifugeTests       | doAll2DTests;

% 3D tests
doSedovTests          = doSedovTests            | doAll3DTests;
