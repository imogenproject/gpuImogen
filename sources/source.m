function source(run, fluids, mag, tFraction)
% This function sources the non-conservative terms in the MHD equations like gravitational
% potential and radiation terms. Effectively it provides the means to add terms that cannot be 
% brought within the del operator, which is the foundation of the spatial fluxing routines.
%
%>< run         data manager object.                                            ImogenManager
%>< mass        mass density                                                    FluidArray  
%>< mom         momentum density                                                FluidArray(3)
%>< ener        energy density                                                  FluidArray
%>< mag         magnetic field density                                          FluidArray(3)

% We observe that { radiation, self-gravity, other pure-mechanical forces } commute:
% radiation acts only on internal energy,
% while purely mechanical forces act only on kinetic energy + momentum
% and self-gravity force is a function only of density

% We double the dtime here because one source step is sandwiched between two flux steps
dTime = tFraction * run.time.dTime;

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
end

% NOTE that this tests the new combined source calculator for frame rotation and gravity field
% This call solves them simultaneously using the implicit midpoint rule with 3 Jacobi iterations
% to converge the predictor step
[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

cudaTestSourceComposite(fluids, run.potentialField.field, run.frameTracking.omega, dTime, run.geometry, run.fluid(1).MINMASS, run.fluid(1).MINMASS*0,  xyvector);

    % FIXME: This could be improved by calculating this affine transform once and storing it
%    if run.frameTracking.omega ~= 0
%        [uv, vv, ~] = run.geometry.ndgridVecs('pos');
%        xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1); 
%        cudaSourceRotatingFrame(fluids, run.frameTracking.omega, dTime/2, xyvector);
%    end
% FIXME does cudaSourceRotatingFrame actually know what to do with multiple fluids?

    for n = 1:numel(run.selfGravity.compactObjects)
        % Applies gravitational sourcing from all compact objects to my domain,
        % And performs a simple accretion action if any part of their zone is in my domain.
        % Need to track star's
        % position (3D), momentum (3D), angular momentum (3D), mass (1D), radius (1D), vaccum_rho(1D), grav_rho(1D), vacccum_E(1D) = 14 doubles
        % Store [X Y Z R Px Py Pz Lx Ly Lz M rhoV rhoG EV] in full state vector:
        lowleft = run.geometry.cornerIndices();
        mass = fluids(1).mass;
        mom = fluids(1).mom;
        ener = fluids(1).ener;
        run.selfGravity.compactObjects{n}.incrementDelta( cudaAccretingStar(mass, mom(1), mom(2), mom(3), ener, run.selfGravity.compactObjects{n}.stateVector, lowleft, run.geometry.d3h(1), dTime, geometry.topology.nproc) );
    end

    %--- External scalar potential (e.g. non self gravitating component) ---%
%    if run.potentialField.ACTIVE
%        cudaSourceScalarPotential(fluids, run.potentialField.field, dTime, run.geometry, run.fluid(1).MINMASS, run.fluid(1).MINMASS * 0);
%    end
    
%    if run.frameTracking.omega ~= 0
%        cudaSourceRotatingFrame(fluids, run.frameTracking.omega, dTime/2, xyvector);
%        clear xyvector;
%    end

if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
    cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
end

    cudaSourceVTO(fluids, [dTime 1 1]);
    %--- Gravitational Potential Sourcing ---%
    %       If the gravitational portion of the code is active, the gravitational potential terms
    %       in both the momentum and energy equations must be appended as source terms.
% Disabling this because SG is dead until further notice
%    if run.selfGravity.ACTIVE
%        enerSource = zeros(run.geometry.localDomainRez);
%        for i=1:3
%            momSource       = dTime*mass.thresholdArray ...
%                                                    .* grav.calculate5PtDerivative(i,run.DGRID{i});
%            enerSource      = enerSource + momSource .* mom(i).array ./ mass.array; % FIXME this fails to identically conserve energy
%            mom(i).array    = mom(i).array - momSource;
%        end
%        ener.array          = ener.array - enerSource;
%    end

    % The mechanical routines above are written to exactly conserve internal energy so that
    % they commute with things which act purely on internal energy (e.g. radiation)
    %--- Radiation Sourcing ---%
    %       If radiation is active, subtract from internal energy
    for N = 1:numel(fluids)
        fluids(N).radiation.solve(fluids(N), mag, dTime);
    end

    if run.selfGravity.ACTIVE || run.potentialField.ACTIVE
        for N = 1:numel(fluids);
	    fluids(N).synchronizeHalos(1, [0 1 1 1 1]);
	    fluids(N).synchronizeHalos(2, [0 1 1 1 1]);
	    fluids(N).synchronizeHalos(3, [0 1 1 1 1]);
	end
    end

% can we keep track of what of these we actually need FIXME
    for N = 1:numel(fluids)
        fluids(N).setBoundaries(0);
    end
end
