function src2f_cmp_2f(run, fluids, mag, tFraction)
% This function handles
% * 2-fluid drag
% * in a rotating frame in cylindrical coordinates with gravity

dTime = run.time.dTime * tFraction;

[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.geometry.frameRotationCenter(1)) (vv-run.geometry.frameRotationCenter(2)) ], 1);

if run.potentialField.ACTIVE
    potOrder = 4;
else
    potOrder = 0;
end

% This call solves geometric source terms, frame rotation and gravity simultaneously
% parameter vector:
% [rho_no gravity, rho_full gravity, omega, dt, space order, time order]
% It can be programmed to use either implicit midpoint (IMP), Runge-Kutta 4 (RK4), Gauss-Legendre 4 (GL4), or GL6

cudaSource2FluidDrag(fluids, run.geometry, [dTime/2, run.multifluidDragMethod]);
cudaTestSourceComposite(fluids, run.potentialField.field, run.geometry, ...
    [run.fluid(1).MINMASS*4, run.fluid(1).MINMASS*4.1, dTime, run.compositeSrcOrders],  xyvector);
cudaSource2FluidDrag(fluids, run.geometry, [dTime/2, run.multifluidDragMethod]);
% Take care of any parallel synchronization we need to do to remain self-consistent

if potOrder > 0
    for N = 1:numel(fluids)
        fluids(N).synchronizeHalos(1, [0 1 1 1 1]);
        fluids(N).synchronizeHalos(2, [0 1 1 1 1]);
        fluids(N).synchronizeHalos(3, [0 1 1 1 1]);
    end
end

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end

end
