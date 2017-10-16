function src1001(run, fluids, mag, tFraction)
% This function sources 2-fluid drag and cylindrical coordinates

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

sigma_gas  = fluids(1).particleSigma;
mu_gas     = fluids(1).particleMu;
sigma_dust = fluids(2).particleSigma;
mu_dust    = fluids(2).particleMu;

% Utilize standard (A/2)(B)(A/2) operator split to acheive 2nd order time accuracy in the
% split terms 
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
cudaSource2FluidDrag(fluids, run.geometry, [sigma_gas, mu_gas, dia_dust, mass_dust, dTime, run.multifluidDragMethod]);
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
