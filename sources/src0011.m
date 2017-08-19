function src0011(run, fluids, mag, tFraction)
% This function sources 2-fluid drag in the presence of gravity

dTime = run.time.dTime * tFraction;

sigma_gas = 2.4e-19;
mu_gas    = 3.3e-27;
dia_dust  = 10e-6; % 10um iron spheres
mass_dust = 3e-11;

cudaSource2FluidDrag(fluids, [sigma_gas, mu_gas, dia_dust, mass_dust, dTime/2]);
cudaSourceScalarPotential(fluids, run.potentialField.field, dTime, run.geometry, run.fluid(1).MINMASS, run.fluid(1).MINMASS * 0);
cudaSource2FluidDrag(fluids, [sigma_gas, mu_gas, dia_dust, mass_dust, dTime/2]);

% Take care of any parallel synchronization we need to do to remain self-consistent
for N = 1:numel(fluids);
    fluids(N).synchronizeHalos(1, [0 1 1 1 1]);
    fluids(N).synchronizeHalos(2, [0 1 1 1 1]);
    fluids(N).synchronizeHalos(3, [0 1 1 1 1]);
end

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end

end
