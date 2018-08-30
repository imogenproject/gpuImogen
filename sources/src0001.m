function src0001(run, fluids, mag, tFraction)
% This function sources 2-fluid drag

dTime = run.time.dTime * tFraction;

cudaSource2FluidDrag(fluids, run.geometry, [dTime, run.multifluidDragMethod]);

if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, run.magnet, dTime);
end

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end

end
