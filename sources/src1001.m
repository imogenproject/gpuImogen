function src1001(run, fluids, mag, tFraction)
% This function sources 2-fluid drag and cylindrical coordinates

dTime = run.time.dTime * tFraction;

% Utilize standard (A/2)(B)(A/2) operator split to acheive 2nd order time accuracy in the
% split terms 
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, run.magnet, dTime/2);
end
cudaSource2FluidDrag(fluids, run.geometry, [dTime, run.multifluidDragMethod]);
if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, run.magnet, dTime/2);
end
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
