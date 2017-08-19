function src1001(run, fluids, mag, tFraction)
% This function sources 2-fluid drag and cylindrical coordinates

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

sigma_gas = 2.4e-19;
mu_gas    = 3.3e-27;
dia_dust  = 10e-6; % 10um iron spheres
mass_dust = 3e-11;

% Utilize standard (A/2)(B)(A) operator split to acheive 2nd order time accuracy in the
% split terms 
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
cudaSource2FluidDrag(fluids, [sigma_gas, mu_gas, dia_dust, mass_dust, dTime]);
cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end


end
