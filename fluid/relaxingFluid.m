function relaxingFluid(run, mass, mom, ener, mag, grav, X)
%   This routine is responsible for the actual fluid fluxing. It utilizes the Relaxed, Total 
%   Variation Diminishing (TVD), Monotone Upwind Scheme for Fluid Dynamics (MUSCL) scheme for
%   non-linear fluxing terms. The relaxation technique ensures that the non-linear function
%   velocities are properly made upwind (See Jin & Xin '95, Ui-Li Pen '04).
%
%>< run      data manager object                                                      ImogenManager
%>< mass     mass density gputag (cell)                                                FluidArray
%>< mom      momentum density gputag (cell)                                            FluidArray(3)
%>< ener     energy density gputag (cell)                                              FluidArray
%>< mag      magnetic field (face)                                                    MagnetArray(3)
%>< grav     gravitational potential                                                  GravityArray
%>> X        vector index of current fluxing direction (1,2,or 3)                     int
    %--- Initialize ---%

    fluxFactor = run.time.dTime ./ run.DGRID{X};
    v          = [mass, mom(1), mom(2), mom(3), ener];

    YYY = 1;
    L = [X 2 3]; L(X)=1;
   
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%                   Half-Timestep predictor step (first-order upwind,not TVD)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[pressa freezea] = freezeAndPtot(mass.gputag, ener.gputag, ...
                                 mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
                                 mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
                                 run.GAMMA, run.pureHydro);

[v(1).store.array v(5).store.array v(L(1)+1).store.array v(L(2)+1).store.array v(L(3)+1).store.array] = cudaWstep(mass.gputag, ener.gputag, ...
                   mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
                   mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
                   pressa, freezea, fluxFactor, run.pureHydro);

GPU_free(pressa); GPU_free(freezea);
cudaArrayAtomic(mass.store.gputag, run.fluid.MINMASS, ENUM.CUATOMIC_SETMIN);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%                   Full-Timestep corrector step (second-order relaxed TVD)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

[pressa freezea] = freezeAndPtot(mass.store.gputag, ener.store.gputag, ...
                                mom(L(1)).store.gputag, mom(L(2)).store.gputag, mom(L(3)).store.gputag, ...
                                mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
                                run.GAMMA, run.pureHydro);


cudaTVDStep(mass.store.gputag, ener.store.gputag, ...
            mom(L(1)).store.gputag, mom(L(2)).store.gputag, mom(L(3)).store.gputag, ...
            mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
            pressa, ...
            mass.gputag, ener.gputag, mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
            freezea, fluxFactor, run.pureHydro);

% The CUDA routines directly overwrite the array so we must apply statics manually
mass.applyStatics();
ener.applyStatics();
mom(1).applyStatics();
mom(2).applyStatics();
mom(3).applyStatics();

GPU_free(pressa); GPU_free(freezea);
cudaArrayAtomic(mass.gputag, run.fluid.MINMASS, ENUM.CUATOMIC_SETMIN);

for t = 1:5; v(t).cleanup(); end

end
