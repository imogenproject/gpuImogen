function relaxingFluid(run, mass, mom, ener, mag, X)
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
%>> X        vector index of current fluxing direction (1,2,or 3)                     int
    %--- Initialize ---%

    fluxFactor = .5*run.time.dTime ./ run.DGRID{X};
    v          = [mass, mom(1), mom(2), mom(3), ener];

    YYY = 1;
    L = [X 2 3]; L(X)=1;
   
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%                   Half-Timestep predictor step (first-order upwind,not TVD)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root)
cs0 = sqrt(run.GAMMA*(run.fluid.MINMASS^(run.GAMMA-1)) );

% freezeAndPtot enforces a minimum pressure
[pressa freezea] = freezeAndPtot(mass.gputag, ener.gputag, ...
                                 mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
                                 mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
                                 run.GAMMA, run.pureHydro, cs0);

GIS = GlobalIndexSemantics();
hostarray = GPU_cudamemcpy(freezea);
mpi_dimreduce(hostarray,X-1,GIS.topology);
GPU_free(freezea);
freezea = GPU_cudamemcpy(hostarray);

% W flux enforces minimum mass density (if using new version, then must pass run.fluid.MINMASS as 13th argument)
[v(1).store.array v(5).store.array v(L(1)+1).store.array v(L(2)+1).store.array v(L(3)+1).store.array] = cudaFluidW(mass.gputag, ener.gputag, ...
                   mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
                   mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
                   pressa, freezea, fluxFactor, run.pureHydro, [run.GAMMA run.fluid.MINMASS]);

%testCF = GPU_cudamemcpy(freezea);
%testP  = GPU_cudamemcpy(pressa);

%GPU_free(pressa);
%GPU_free(freezea);
%cudaArrayAtomic(mass.store.gputag, run.fluid.MINMASS, ENUM.CUATOMIC_SETMIN);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%                   Full-Timestep corrector step (second-order relaxed TVD)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%[pressa freezea] = freezeAndPtot(mass.store.gputag, ener.store.gputag, ...
%                                mom(L(1)).store.gputag, mom(L(2)).store.gputag, mom(L(3)).store.gputag, ...
%                                mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
%                                run.GAMMA, run.pureHydro, cs0);

%testCF2 = GPU_cudamemcpy(freezea) - testCF;
%testP2  = GPU_cudamemcpy(pressa)  - testP;

%fprintf('max dcf: %g; min dcf: %g; max dp: %g; min dp: %g\n', max(testCF2(:)), min(testCF2(:)), max(testP2(:)), min(testP2(:)) );

GIS = GlobalIndexSemantics();
hostarray = GPU_cudamemcpy(freezea);
mpi_dimreduce(hostarray,X-1,GIS.topology);
GPU_free(freezea);
freezea = GPU_cudamemcpy(hostarray);

cudaFluidTVD(mass.store.gputag, ener.store.gputag, ...
            mom(L(1)).store.gputag, mom(L(2)).store.gputag, mom(L(3)).store.gputag, ...
            mag(L(1)).cellMag.gputag, mag(L(2)).cellMag.gputag, mag(L(3)).cellMag.gputag, ...
            pressa, ...
            mass.gputag, ener.gputag, mom(L(1)).gputag, mom(L(2)).gputag, mom(L(3)).gputag, ...
            freezea, fluxFactor, run.pureHydro);%, [run.fluid.MINMASS, run.GAMMA]);

% The CUDA routines directly overwrite the array so we must apply statics manually
mass.applyStatics();
ener.applyStatics();
mom(1).applyStatics();
mom(2).applyStatics();
mom(3).applyStatics();

GPU_free(pressa);
GPU_free(freezea);
cudaArrayAtomic(mass.gputag, run.fluid.MINMASS, ENUM.CUATOMIC_SETMIN); % this is handled by the fluid calls now

for t = 1:5; v(t).cleanup(); end % Delete upwind storage arrays

end
