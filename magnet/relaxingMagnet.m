function relaxingMagnet(run, mag, velGrid, X, I)
% Flux the input magnetic field according to the two-step MUSCL TVD scheme and return the result as 
% well as the calculated flux to use later for the constraint quantity according to constrained 
% transport.
%
%>< run                  run variable manager object                                      ImogenManager
%>< mag                  magnetic field array (3D)                                        MagnetArray(3)
%>< velGrid              face aligned velocity array                                      FluidArray
%>> X                    vector index for fluxing direction                               int
%>> I                    the component of the b-field to operate on                       int

    %-----------------------------------------------------------------------------------------------
    % Initialization
    %---------------
    fluxFactor = 0.5*run.time.dTime ./ run.geometry.d3h(X);

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Half-Timestep predictor step (first-order upwind,not TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    [mag(I).store(X).array, velocityFlow] = cudaMagW(mag(I), velGrid, fluxFactor, X);
%saveDEBUG(mag(I).store(X).array,sprintf('magI w after w upwind'));

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Full-Timestep corrector step (second-order relaxed TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    fluxFactor = 2*fluxFactor; %Multiply to get full timestep

    % FIXME: fix this braindead hack
    magflux = mag(I).flux(X);
    [magflux.array] = cudaMagTVD(mag(I).store(X), mag(I), velGrid, fluxFactor, X);
    % This returns the flux array, pre-shifted forward one in the X direction to avoid the shift originally present below
    cudaHaloExchange(mag(I),  [1 2 3], X, run.geometry.topology, mag(I).bcHaloShare);
    cudaHaloExchange(magflux, [1 2 3], X, run.geometry.topology, mag(I).bcHaloShare);

    mag(I).applyBoundaryConditions(X);
    
    %-----------------------------------------------------------------------------------
    % Reuse advection flux for constraint step for CT
    %------------------------------------------------
    fluxFactor = run.time.dTime ./ run.geometry.d3h(I);

    cudaFwdDifference(mag(X), magflux, I, fluxFactor);
    mag(X).applyBoundaryConditions(I);
    GPU_free(velocityFlow);

    % FIXME: fix this braindead hack
%    ec = double([strcmp(mag(X).bcModes{1,I},'circ'); strcmp(mag(X).bcModes{2,I},'circ')]);
    cudaHaloExchange(mag(X), [1 2 3], I, run.geometry.topology,mag(X).bcHaloShare);
end
