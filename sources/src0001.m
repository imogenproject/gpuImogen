function src0001(run, fluids, mag, tFraction)
% This function sources 2-fluid drag

dTime = run.time.dTime * tFraction;

sigma_gas  = fluids(1).particleSigma;
mu_gas     = fluids(1).particleMu;
sigma_dust = fluids(2).particleSigma;
mu_dust    = fluids(2).particleMu;

cudaSource2FluidDrag(fluids, [sigma_gas, mu_gas, sigma_dust, mu_dust, dTime]);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end

end
