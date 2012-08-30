function bicgstabPotentialSolverIni(manager, mass)
% This function initializes the bicgstab solver; It prepares the coefficient matrix and
% the preconditioner
%
%>< run         Data manager                                                    ImogenManager
%>< mass        Mass density                                                    FluidArray
%>< grav        Gravitational potential                                         GravityArray

manager.createSparseMatrix(mass.gridSize, [run.DGRID{1} run.DGRID{2} run.DGRID{3}]);

manager.array = zeros(mass.gridSize);

end
