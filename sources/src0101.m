function src0101(run, fluids, mag, tFraction)
% This function sources 2-fluid drag in a rotating frame

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

cudaSourceRotatingFrame(fluids, run.geometry.frameOmega, dTime/2, xyvector);
if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, run.magnet, dTime/2);
end
cudaSource2FluidDrag(fluids, run.geometry, [dTime, run.multifluidDragMethod]);
if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, run.magnet, dTime/2);
end
cudaSourceRotatingFrame(fluids, run.geometry.frameOmega, dTime/2, xyvector);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
