function relaxingFluid(run, mass, mom, ener, mag, X)
%   This routine is responsible for the actual fluid fluxing. It utilizes the Relaxed, Total 
%   Variation Diminishing (TVD), Monotone Upwind Scheme for Fluid Dynamics (MUSCL) scheme for
%   non-linear fluxing terms. The relaxation technique ensures that the non-linear function
%   velocities are properly made upwind (See Jin & Xin '95, Ui-Li Pen '04).
%
%>< run      data manager object                                                ImogenManager
%>< mass     mass density (cell)                                                FluidArray
%>< mom      momentum density (cell)                                            FluidArray(3)
%>< ener     energy density (cell)                                              FluidArray
%>< mag      magnetic field (face)                                              MagnetArray(3)
%>> X        vector index of current fluxing direction (1,2,or 3)               int

fluxFactor = run.time.dTime ./ run.DGRID{X};
v          = [mass, mom(1), mom(2), mom(3), ener];
   
GIS = GlobalIndexSemantics();

% Advance fluid quantities through a 2nd order upwind timestep
% third [] parameter: 1 = HLL, 2 = HLLC, 3 = Xin/Jin
% 4th   []          : flux direction
% If debug, dbgoutput = cudaFluidStep(...)
cudaFluidStep([fluxFactor, run.pureHydro, run.GAMMA run.fluid(1).MINMASS run.cfdMethod X], ...
              mass, ener, mom(1), mom(2), mom(3), mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, GIS.topology);

% Must call applyStatics because the cudaFluidTVD call overwrites the array directly.
for t = 1:5; v(t).applyBoundaryConditions(X); v(t).cleanup(); end % Delete upwind storage arrays

end

