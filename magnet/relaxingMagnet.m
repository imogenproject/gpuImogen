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
    fluxFactor = 0.5*run.time.dTime ./ run.DGRID{X};
    GIS = GlobalIndexSemantics();

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Half-Timestep predictor step (first-order upwind,not TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    [mag(I).store(X).array velocityFlow] = cudaMagW(mag(I).gputag, velGrid.gputag, fluxFactor, X);

    velocityFlow = GPU_Type(velocityFlow);

%saveDEBUG(mag(I).store(X).array,sprintf('magI w after w upwind'));

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Full-Timestep corrector step (second-order relaxed TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    fluxFactor = 2*fluxFactor; %Multiply to get full timestep

    % FIXME: fix this braindead hack
%    ec = double([strcmp(mag(I).bcModes{1,X},'circ'); strcmp(mag(I).bcModes{2,X},'circ')]);
    [mag(I).flux(X).array] = cudaMagTVD(mag(I).store(X).gputag, mag(I).gputag, velGrid.gputag, velocityFlow.GPU_MemPtr, fluxFactor, X);

%    cudaHaloExchange(mag(I).gputag,         [1 2 3], X, GIS.topology, GIS.edgeInterior(:,X));
    cudaHaloExchange(mag(I).gputag,         [1 2 3], X, GIS.topology, mag(I).bcHaloShare);
    cudaHaloExchange(mag(I).flux(X).gputag, [1 2 3], X, GIS.topology, mag(I).bcHaloShare);


    mag(I).applyStatics();
    % This returns the flux array, pre-shifted forward one in the X direction to avoid the shift originally present below
    
    %-----------------------------------------------------------------------------------
    % Reuse advection flux for constraint step for CT
    %------------------------------------------------
    fluxFactor = run.time.dTime ./ run.DGRID{I};

    cudaFwdDifference(mag(X).gputag, mag(I).flux(X).gputag, I, fluxFactor);
    mag(X).applyStatics();

    % FIXME: fix this braindead hack
%    ec = double([strcmp(mag(X).bcModes{1,I},'circ'); strcmp(mag(X).bcModes{2,I},'circ')]);
    cudaHaloExchange(mag(X).gputag, [1 2 3], I, GIS.topology,mag(X).bcHaloShare);


end
