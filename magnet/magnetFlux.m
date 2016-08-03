function magnetFlux(run, mass, mom, mag, X, magneticIndices)
%   This is the routine responsible for fluxing the magnetic field array according to a TVD MUSCL
%   scheme that utilizes constrained transport to maintain the divergencelessness of the magnetic
%   field over time if it is divergenceless as an initial condition.
%
%>< run                 Run manager object.                                     ImogenManager
%>< mass                Mass density array.                                     FluidArray
%>< mom                 Momentum density array.                                 FluidArray(3)
%>< ener                Energy density array.                                   FluidArray
%>< mag                 Magnetic field array.                                   MagnetArray(3)
%>> X                   Vector index along which to flux.                       int
%>> magneticIndices     Index values for the magnetic components to flux        int(2)
    for n=1:2
        I = magneticIndices(n); % Component of the b-field to flux.

        % Prepare arrays for magnetic flux step at cell corners (2nd order grid-aligned velocity)
	mag(I).velGrid(X).array = cudaMagPrep(mom(X), mass, [I X]);

%cudaHaloExchange(mag(I).velGrid(X).gputag, [1 2 3], X, run.geometry.topology, GIS.edgeInterior(:,X));
        cudaHaloExchange(mag(I).velGrid(X), [1 2 3], X, run.geometry.topology, mag(I).bcHaloShare);
        cudaHaloExchange(mag(I).velGrid(X), [1 2 3], I, run.geometry.topology, mag(I).bcHaloShare);

        % Flux Advection step
        relaxingMagnet(run, mag, mag(I).velGrid(X), X, I);
        mag(I).cleanup();

    end

    for n = 1:3
        mag(n).updateCellCentered();
 %       mag(X).updateCellCentered(); % isn't this right?
    end

%    cudaHaloExchange(mag(X).cellMag.gputag, [1 2 3], I, run.geometry.topology, GIS.edgeInterior(:,I));
%    cudaHaloExchange(mag(X).cellMag.gputag, [1 2 3], X, run.geometry.topology, GIS.edgeInterior(:,X));
%    cudaHaloExchange(mag(I).cellMag.gputag, [1 2 3], I, run.geometry.topology, GIS.edgeInterior(:,I));
%    cudaHaloExchange(mag(I).cellMag.gputag, [1 2 3], X, run.geometry.topology, GIS.edgeInterior(:,X));

end
