function srcBlank(run, fluids, mag, tFraction)
% This function sources nothing (except maybe radiation)

dTime = run.time.dTime * tFraction;

if run.radiation.active
    run.radiation.opticallyThinSolver(fluids, mag, dTime);
end

end
