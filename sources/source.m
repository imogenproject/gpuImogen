function source(run, mass, mom, ener, mag)
% This function sources the non-conservative terms in the MHD equations like gravitational potential
% and radiation terms. Effectively it provides the means to add terms that cannot be brought within 
% the del operator, which is the foundation of the spatial fluxing routines.
%
%>< run			data manager object.                                            ImogenManager
%>< mass		mass density                                                    FluidArray  
%>< mom			momentum density                                                FluidArray(3)
%>< ener        energy density                                                  FluidArray
%>< mag         magnetic field density                                          FluidArray(3)

    %--- External scalar potential (e.g. non self gravitating component) ---%
    if run.potentialField.ACTIVE
        cudaSourceScalarPotential(mass.gputag, ener.gputag, mom(1).gputag, mom(2).gputag, mom(3).gputag, run.potentialField.field.GPU_MemPtr, run.time.dTime, [run.DGRID{1} run.DGRID{2} run.DGRID{3}], run.fluid.MINMASS*ENUM.GRAV_FEELGRAV_COEFF);
    end

    % TESTING
%    GIS = GlobalIndexSemantics();
%    [xg yg] = GIS.ndgridVecs;
%    xg = GPU_Type(run.DGRID{1}*(xg-128.5));
%    yg = GPU_Type(run.DGRID{2}*(yg-128.5));
%    cudaSourceRotatingFrame(mass.gputag, ener.gputag, mom(1).gputag, mom(2).gputag, 1, run.time.dTime, xg.GPU_MemPtr, yg.GPU_MemPtr);
%clear xg
%clear yg

    %--- Gravitational Potential Sourcing ---%
    %       If the gravitational portion of the code is active, the gravitational potential terms
    %       in both the momentum and energy equations must be appended as source terms.
    if run.selfGravity.ACTIVE
        enerSource = zeros(run.gridSize);
        for i=1:3
            momSource       = run.time.dTime*mass.thresholdArray ...
                                                    .* grav.calculate5PtDerivative(i,run.DGRID{i});
            enerSource      = enerSource + momSource .* mom(i).array ./ mass.array;
            mom(i).array    = mom(i).array - momSource;
        end
        ener.array          = ener.array - enerSource;
    end
    
    %--- Radiation Sourcing ---%
    %       If radiation is active, the radiation terms are subtracted, as a sink, from the energy
    %       equation.
    if run.fluid.radiation.type ~= ENUM.RADIATION_NONE
        ener.array              = ener.array - run.time.dTime*run.fluid.radiation.solve(run, mass, mom, ener, mag);
    end

    if run.potentialField.ACTIVE | run.selfGravity.ACTIVE | (run.fluid.radiation.type ~= ENUM.RADIATION_NONE)
        % Oh you better believe we need to synchronize up in dis house
        GIS = GlobalIndexSemantics();
        S = {mom(1), mom(2), mom(3), ener};
        for j = 1:4; for dir = 1:3
%            iscirc = double([strcmp(S{j}.bcModes{1,dir},ENUM.BCMODE_CIRCULAR) strcmp(S{j}.bcModes{2,dir}, ENUM.BCMODE_CIRCULAR)]);
            cudaHaloExchange(S{j}.gputag, [1 2 3], dir, GIS.topology, GIS.edgeInterior(:,dir));
        end; end
    end
    
end
