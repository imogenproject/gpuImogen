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

%if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
[uv, vv, ~] = run.geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);
%    cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
%end

% This call solves geometric source terms, frame rotation and gravity simultaneously
% It can be programmed to use either implicit midpoint (IMP), Runge-Kutta 4 (RK4) or Gauss-Legendre 4 (GL4)
cudaTestSourceComposite(fluids, run.potentialField.field, run.geometry, ...
    [run.fluid(1).MINMASS*4, run.fluid(1).MINMASS*4.1, run.frameTracking.omega, dTime, 2, 2],  xyvector);

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

%if run.geometry.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
%    cudaSourceCylindricalTerms(fluids, dTime/2, run.geometry);
%end

    % The "vacuum taffy operator" serves mainly to keep the false vacuum from accelerating
    % without limit under the influence of gravity and blowtorching a simulation volume
    % with infalling gas. This isn't physical so we disregard normal accuracy and symmetry considerations
    % Rotation rate and geometry are necessary to decay to the proper values in noninertial frames.
    % Scalar vector:
    %    [dtime, velocity decay rate, density decay rate, omega]
    % v/rho decay rates have dimension 1/time; larger #s make the VTO more aggressive
    
    % pi/2 rate sets characteristic time to 2MIRPs
    % rho/T floor is instant/fixed
    cudaSourceVTO(fluids, [dTime pi/2 pi/2 run.frameTracking.omega], run.geometry);

    % The mechanical routines above are written to exactly conserve internal energy so that
    % they commute with things which act purely on internal energy (e.g. radiation)
    %--- Radiation Sourcing ---%
    %       If radiation is active, subtract from internal energy
    for N = 1:numel(fluids)
        fluids(N).radiation.solve(fluids(N), mag, dTime);
    end

    % Take care of any parallel synchronization we need to do to remain self-consistent
    if run.selfGravity.ACTIVE || run.potentialField.ACTIVE
        for N = 1:numel(fluids);
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
