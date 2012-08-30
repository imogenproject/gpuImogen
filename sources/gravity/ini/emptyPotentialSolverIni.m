function emptyPotentialSolverIni(manager, mass)
% This function handles the initialization process for non-gravitating cases by setting the 
% gravitational potential array to zero.
%
%>< run     Data manager object.                                            ImogenManager
%>< mass    Mass density. object                                            FluidArray
%>< grav    Gravitational potential array                                   GravityArray

    manager.array = 0;
end
