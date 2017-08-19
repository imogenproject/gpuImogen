function src0010(run, fluids, mag, tFraction)
% This function sources gravity potential only

dTime = run.time.dTime * tFraction;

cudaSourceScalarPotential(fluids, run.potentialField.field, dTime, run.geometry, run.fluid(1).MINMASS, run.fluid(1).MINMASS * 0);

% Take care of any parallel synchronization we need to do to remain self-consistent
for N = 1:numel(fluids);
    fluids(N).synchronizeHalos(1, [0 1 1 1 1]);
    fluids(N).synchronizeHalos(2, [0 1 1 1 1]);
    fluids(N).synchronizeHalos(3, [0 1 1 1 1]);
end

% Assert boundary conditions
for N = 1:numel(fluids)
    fluids(N).setBoundaries(1);
    fluids(N).setBoundaries(2);
    fluids(N).setBoundaries(3);
end

end
