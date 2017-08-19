function source0100(run, fluids, mag, tFraction)
% This function sources rotating frame terms

dTime = tFraction * run.time.dTime;

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    [uv, vv, ~] = run.geometry.ndgridVecs('pos');
    xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);
    cudaSourceRotatingFrame(fluids, run.frameTracking.omega, dTime, xyvector);
end

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
