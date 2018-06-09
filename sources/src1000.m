function src1000(run, fluids, mag, tFraction)
% This function sources cylindrical geometry terms

dTime = tFraction * run.time.dTime;

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
