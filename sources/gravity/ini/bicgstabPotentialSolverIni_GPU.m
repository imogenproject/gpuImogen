function bicgstabPotentialSolverIni_GPU(manager, mass)
% This function initializes the bicgstab solver; It prepares the coefficient matrix and
% the preconditioner
%
%>< run         Data manager                                                    ImogenManager
%>< mass        Mass density                                                    FluidArray
%>< grav        Gravitational potential                                         GravityArray

%run.gravity.createSparseMatrix(mass.gridSize, [run.DGRID{1} run.DGRID{2} run.DGRID{3}]);

grav.array = GPU_Type(zeros(mass.gridSize));

end
