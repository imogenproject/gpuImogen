function src0100(run, fluids, mag, tFraction)
% This function sources rotating frame terms

dTime = tFraction * run.time.dTime;

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    [uv, vv, ~] = run.geometry.ndgridVecs('pos');
    xyvector = GPU_Type([ (uv-run.geometry.frameRotationCenter(1)) (vv-run.geometry.RotationCenter(2)) ], 1);
    cudaSourceRotatingFrame(fluids, run.geometry.frameOmega, dTime, xyvector);
end

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
