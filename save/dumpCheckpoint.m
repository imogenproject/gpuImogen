function backup = dumpCheckpoint(run)

if mpi_amirank0()
    fprintf('NOTE: Dumping checkpoint at iteration %i.\n', run.time.iteration);
end

backup.iteration = run.time.iteration;
backup.oldTime   = run.time.time;
backup.oldCFL    = run.time.CFL;
backup.arrays    = cell([5*numel(run.fluid) 1]);

ctr = 1;
for a = 1:numel(run.fluid)
    DH = run.fluid(a).DataHolder;
    
    for b = 0:4
        backup.arrays{ctr} = GPU_download(GPU_getslab(DH, b));
        ctr = ctr + 1;
    end
end

end
