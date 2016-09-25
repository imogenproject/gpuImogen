function relaxingFluid(run, afluid, mag, X)
%   This routine is responsible for the actual fluid fluxing. It utilizes the Relaxed, Total 
%   Variation Diminishing (TVD), Monotone Upwind Scheme for Fluid Dynamics (MUSCL) scheme for
%   non-linear fluxing terms. The relaxation technique ensures that the non-linear function
%   velocities are properly made upwind (See Jin & Xin '95, Ui-Li Pen '04).
%
%>< run      data manager object                                                ImogenManager
%>< afluid   FLuidManager object                                                FluidManager
%>< mass     mass density (cell)                                                FluidArray
%>< mom      momentum density (cell)                                            FluidArray(3)
%>< ener     energy density (cell)                                              FluidArray
%>< mag      magnetic field (face)                                              MagnetArray(3)
%>> X        vector index of current fluxing direction (1,2,or 3)               int

dt = run.time.dTime;
v  = [afluid.mass, afluid.mom(1), afluid.mom(2), afluid.mom(3), afluid.ener];

% Advance fluid quantities through a 2nd order upwind timestep
% 1st parameter : timestep
% 2nd parameter : whether to assume pure hydro (ignore mag)
% 3rd parameter : adiabatic index
% 4th parameter : minimum density to enforce
% 5th parameter : 1 = HLL, 2 = HLLC, 3 = Xin/Jin
% 6th parameter : direction (123 = XYZ)
% If debug, dbgoutput = cudaFluidStep(...)
params = [run.time.dTime, run.pureHydro, afluid.gamma, afluid.MINMASS, run.cfdMethod, X];

cudaFluidStep(params, afluid.mass, afluid.ener, afluid.mom(1), afluid.mom(2), ...
              afluid.mom(3), mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, run.geometry);

% Must call applyStatics because the cudaFluidTVD call overwrites the array directly.
for t = 1:5; v(t).applyBoundaryConditions(X); v(t).cleanup(); end % Delete upwind storage arrays

end

