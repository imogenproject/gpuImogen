function relaxingMagnet(run, mag, velGrid, X, I)
% Flux the input magnetic field according to the two-step MUSCL TVD scheme and return the result as 
% well as the calculated flux to use later for the constraint quantity according to constrained 
% transport.
%
%><	run				run variable manager object								ImogenManager
%>< mag             magnetic field array (3D)								MagnetArray(3)
%>< velGrid         face aligned velocity array								FluidArray
%>> X				vector index for fluxing direction						int
%>>	I				the component of the b-field to operate on				int

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

%if X == 1
%disp('gpu routine used');
    [mag(I).flux(X).array] = cudaMagTVD(mag(I).store(X).gputag, mag(I).gputag, velGrid.gputag, velocityFlow.GPU_MemPtr, fluxFactor, X);
%else
%disp('normal routines used');
%    mag(I).wMag(X).array = mag(I).store(X).array .* velGrid.array;
%    
%    mag(I).flux(X).array = mag(I).wMag(X).array .* (1-velocityFlow.array) ...
%                           + mag(I).wMag(X).shift(X,1).array .* velocityFlow.array;
%    dFluxR  = ( mag(I).wMag(X).shift(X,1).array - mag(I).flux(X).array ) .* (1-velocityFlow.array) ...
%            + ( mag(I).flux(X).array - mag(I).wMag(X).shift(X,2).array ) .* velocityFlow.array;
%    dFluxL  = ( mag(I).flux(X).array - mag(I).wMag(X).shift(X,-1).array ) .* (1-velocityFlow.array) ...
%           + ( mag(I).wMag(X).array - mag(I).flux(X).array ) .* velocityFlow.array;
%   run.magnet.limiter{X}(mag(I).flux(X), dFluxL, dFluxR); % This is doubled, appropriate halving done by limiter functions

%if max(max(abs(mag(I).flux(X).array))) > .5
%  error('wtf flux')
%end

%   mag(I).array = mag(I).array - fluxFactor .* ( mag(I).flux(X).array - mag(I).flux(X).shift(X,-1).array );
%end
    
    %-----------------------------------------------------------------------------------
    % Reuse advection flux for constraint step for CT
    %------------------------------------------------
    fluxFactor = run.time.dTime ./ run.DGRID{I};

    mag(I).flux(X).array = mag(I).flux(X).array - mag(I).flux(X).shift(I,1).array;
    mag(I).flux(X).array =  mag(I).flux(X).shift(X,-1);
    mag(X).array = mag(X).array - fluxFactor .* mag(I).flux(X).array;
end
