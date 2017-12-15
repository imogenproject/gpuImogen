function source1000(run, fluids, mag, tFraction)
% This function sources cylindrical geometry terms

dTime = tFraction * run.time.dTime;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
%xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);
cudaSourceCylindricalTerms(fluids, dTime, run.geometry);

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
