function multigridPotentialSolverIni(manager, mass)
% This function initializes the multigrid potential solver; It generates and stores the finest
% level position data so that the solver doesn't need to recompute it every time
%
%>< run         Data manager                                                ImogenManager
%>< mass        Mass density                                                FluidArray
%>< grav        Gravitational potential                                     GravityArray
% FIXME broken in parallel oh lord so badly broken
    [rho, poss]           = massQuantization(mass.array, run.geometry.localDomainRez, run.geometry.d3h);
    nlevels               = numel(poss);
    manager.MG_TOPPOS = poss{nlevels};
end
