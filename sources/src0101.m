function src0101(run, fluids, mag, tFraction)
% This function sources 2-fluid drag in a rotating frame

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

sigma_gas  = fluids(1).particleSigma;
mu_gas     = fluids(1).particleMu;
sigma_dust = fluids(2).particleSigma;
mu_dust    = fluids(2).particleMu;

cudaSourceRotatingFrame(fluids, run.frameTracking.omega dTime/2, xyvector);
cudaSource2FluidDrag(fluids, [sigma_gas, mu_gas, dia_dust, mass_dust, dTime]);
cudaSourceRotatingFrame(fluids, run.frameTracking.omega, dTime/2, xyvector);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
