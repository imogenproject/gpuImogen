function src0101(run, fluids, mag, tFraction)
% This function sources 2-fluid drag in a rotating frame

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);


sigma_gas = 2.4e-19;
mu_gas    = 3.3e-27;
dia_dust  = 10e-6; % 10um iron spheres
mass_dust = 3e-11;

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
