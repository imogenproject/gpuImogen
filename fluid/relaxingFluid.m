function relaxingFluid(run, mass, mom, ener, mag, X)
%   This routine is responsible for the actual fluid fluxing. It utilizes the Relaxed, Total 
%   Variation Diminishing (TVD), Monotone Upwind Scheme for Fluid Dynamics (MUSCL) scheme for
%   non-linear fluxing terms. The relaxation technique ensures that the non-linear function
%   velocities are properly made upwind (See Jin & Xin '95, Ui-Li Pen '04).
%
%>< run      data manager object                                                      ImogenManager
%>< mass     mass density (cell)                                                FluidArray
%>< mom      momentum density (cell)                                            FluidArray(3)
%>< ener     energy density (cell)                                              FluidArray
%>< mag      magnetic field (face)                                                    MagnetArray(3)
%>> X        vector index of current fluxing direction (1,2,or 3)                     int
    %--- Initialize ---%

fluxFactor = run.time.dTime ./ run.DGRID{X};
L          = [X 2 3]; L(X)=1;
v          = [mass, mom(L(1)), mom(L(2)), mom(L(3)), ener];
   
% Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root) for adiabatically
% compressed fluid
cs0 = sqrt(run.GAMMA*(run.fluid.MINMASS^(run.GAMMA-1)) );

% freezeAndPtot enforces a minimum pressure
[pressa freezea] = freezeAndPtot(mass, ener, mom(L(1)), mom(L(2)), mom(L(3)), ...
    mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, run.GAMMA, run.pureHydro, cs0);

GIS = GlobalIndexSemantics();

% Advanced fluid quantities through a 2nd order upwind timestep
% third [] parameter: 1 = HLL, 2 = HLLC, 3 = Xin/Jin
% 4th   []          : flux direction
% If debug, dbgoutput = cudaFluidStep(...)
cudaFluidStep(mass, ener, mom(1), mom(2), mom(3), mag(L(1)).cellMag, mag(L(2)).cellMag, ...
    mag(L(3)).cellMag, pressa, freezea, fluxFactor, run.pureHydro, [run.GAMMA run.fluid.MINMASS 2 X], GIS.topology);


GPU_free(pressa);
GPU_free(freezea);

% Must call applyStatics because the cudaFluidTVD call overwrites the array directly.
for t = 1:5; v(t).applyBoundaryConditions(1); v(t).cleanup(); end % Delete upwind storage arrays

end

