function multigridPotentialSolverIni(manager, mass)
% This function initializes the multigrid potential solver; It generates and stores the finest
% level position data so that the solver doesn't need to recompute it every time
%
%>< run         Data manager                                                ImogenManager
%>< mass        Mass density                                                FluidArray
%>< grav        Gravitational potential                                     GravityArray

    [rho, poss]           = massQuantization(mass.array, run.gridSize, run.DGRID);
    nlevels               = numel(poss);
    manager.MG_TOPPOS = poss{nlevels};
end
