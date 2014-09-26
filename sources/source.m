function source(run, mass, mom, ener, mag, tFraction)
% This function sources the non-conservative terms in the MHD equations like gravitational potential
% and radiation terms. Effectively it provides the means to add terms that cannot be brought within 
% the del operator, which is the foundation of the spatial fluxing routines.
%
%>< run			data manager object.                                            ImogenManager
%>< mass		mass density                                                    FluidArray  
%>< mom			momentum density                                                FluidArray(3)
%>< ener        energy density                                                  FluidArray
%>< mag         magnetic field density                                          FluidArray(3)

GIS = GlobalIndexSemantics();

% We observe that { radiation, self-gravity, other pure-mechanical forces } commute:
% radiation acts only on internal energy,
% while purely mechanical forces act only on kinetic energy + momentum
% and self-gravity force is a function only of density

dTime = 2 * tFraction * run.time.dTime;

    % FIXME: This could be improved by calculating this affine transform once and storing it
    if run.frameRotateOmega ~= 0
        [xg yg] = GIS.ndgridVecs;
        xyvector = GPU_Type([ run.DGRID{1}*(xg-run.frameRotateCenter(1)) run.DGRID{2}*(yg-run.frameRotateCenter(2)) ]); 
        cudaSourceRotatingFrame(mass, ener, mom(1), mom(2), run.frameRotateOmega, dTime, xyvector);
        clear xyvector;
    end

    for n = 1:numel(run.selfGravity.compactObjects)
        % Applies gravitational sourcing from all compact objects to my domain,
        % And performs a simple accretion action if any part of their zone is in my domain.
        % Need to track star's
        % position (3D), momentum (3D), angular momentum (3D), mass (1D), radius (1D), vaccum_rho(1D), grav_rho(1D), vacccum_E(1D) = 14 doubles
        % Store [X Y Z R Px Py Pz Lx Ly Lz M rhoV rhoG EV] in full state vector:
        lowleft = GIS.cornerIndices();
        run.selfGravity.compactObjects{n}.incrementDelta( cudaAccretingStar(mass, mom(1), mom(2), mom(3), ener, run.selfGravity.compactObjects{n}.stateVector, lowleft, run.DGRID{1}, dTime, GIS.topology.nproc) );
    end


    %--- External scalar potential (e.g. non self gravitating component) ---%
    if run.potentialField.ACTIVE
        cudaSourceScalarPotential(mass, ener, mom(1), mom(2), mom(3), run.potentialField.field, dTime, [run.DGRID{1} run.DGRID{2} run.DGRID{3}], run.fluid.MINMASS, run.fluid.MINMASS*ENUM.GRAV_FEELGRAV_COEFF);
    end

    % FIXME: This sequence is only first order accurate in the energy equation
    %--- Gravitational Potential Sourcing ---%
    %       If the gravitational portion of the code is active, the gravitational potential terms
    %       in both the momentum and energy equations must be appended as source terms.
% Disabling this because SG is dead until further notice
%    if run.selfGravity.ACTIVE
%        enerSource = zeros(run.gridSize);
%        for i=1:3
%            momSource       = dTime*mass.thresholdArray ...
%                                                    .* grav.calculate5PtDerivative(i,run.DGRID{i});
%            enerSource      = enerSource + momSource .* mom(i).array ./ mass.array;
%            mom(i).array    = mom(i).array - momSource;
%        end
%        ener.array          = ener.array - enerSource;
%    end
    

    % The mechanical routines above are written to exactly conserve internal energy so that
    % they commute with things which act purely on internal energy (e.g. radiation)
    %--- Radiation Sourcing ---%
    %       If radiation is active, subtract from internal energy
    if strcmp(run.fluid.radiation.type, ENUM.RADIATION_NONE) == false
        run.fluid.radiation.solve(run, mass, mom, ener, mag, dTime);
    end

    if run.selfGravity.ACTIVE | run.potentialField.ACTIVE
        % Oh you better believe we need to synchronize up in dis house
        GIS = GlobalIndexSemantics();
        S = {mom(1), mom(2), mom(3), ener};
        for j = 1:4; for dir = 1:3
%            iscirc = double([strcmp(S{j}.bcModes{1,dir},ENUM.BCMODE_CIRCULAR) strcmp(S{j}.bcModes{2,dir}, ENUM.BCMODE_CIRCULAR)]);
            cudaHaloExchange(S{j}.gputag, [1 2 3], dir, GIS.topology, GIS.edgeInterior(:,dir));
        end; end
    end

mass.applyBoundaryConditions(0);
ener.applyBoundaryConditions(0);
mom(1).applyBoundaryConditions(0);
mom(2).applyBoundaryConditions(0);
mom(3).applyBoundaryConditions(0);

end
