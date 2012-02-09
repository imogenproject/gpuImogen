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

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Half-Timestep predictor step (first-order upwind,not TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    [mag(I).store(X).array velocityFlow] = cudaMagW(mag(I).gputag, velGrid.gputag, fluxFactor, X);

    velocityFlow = GPU_Type(velocityFlow);

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Full-Timestep corrector step (second-order relaxed TVD)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    fluxFactor = 2*fluxFactor; %Multiply to get full timestep

    [mag(I).flux(X).array] = cudaMagTVD(mag(I).store(X).gputag, mag(I).gputag, velGrid.gputag, velocityFlow.GPU_MemPtr, fluxFactor, X);
    
    %-----------------------------------------------------------------------------------
    % Reuse advection flux for constraint step for CT
    %------------------------------------------------
    fluxFactor = run.time.dTime ./ run.DGRID{I};

    mag(I).flux(X).array = mag(I).flux(X).array - mag(I).flux(X).shift(I,1).array;
    mag(I).flux(X).array =  mag(I).flux(X).shift(X,-1);
    mag(X).array = mag(X).array - fluxFactor .* mag(I).flux(X).array;
end
