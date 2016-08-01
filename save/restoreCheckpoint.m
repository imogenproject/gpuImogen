function restoreCheckpoint(run, dump)

if mpi_amirank0()
   fprintf('NOTE: Checkpoint restore triggered at iteration %i back to iteration %i.\n', run.time.iteration, dump.iteration);
   fprintf('WARNING: CHeckpoint restore triggered; CFL will now be reduced from %f to %f.\n', run.time.CFL, .75*run.time.CFL);
end

run.time.iteration = dump.iteration;
run.time.time      = dump.oldTime;
run.time.CFL       = 0.75*run.time.CFL;

ctr = 1;
for a = 1:numel(run.fluid)
    DH = run.fluid(a).DataHolder;
    
    rho = dump.arrays{ctr};
    ener= dump.arrays{ctr+1};
    px  = dump.arrays{ctr+2};
    py  = dump.arrays{ctr+3};
    pz  = dump.arrays{ctr+4};
    
    P = (run.GAMMA - 1) * (ener - .5*(px.^2+py.^2+pz.^2)./rho);
    
    dconstant = 0.3;
    
    px = px + dconstant * del2(px);
    py = py + dconstant * del2(py);
    pz = pz + dconstant * del2(pz);
    rho= rho+ dconstant * del2(rho);
    P  = P  + dconstant * del2(P);
    ener = .5*(px.^2+py.^2+pz.^2)./rho + P/(run.GAMMA-1);
    
    dump.arrays{ctr} = rho;
    dump.arrays{ctr+1} = ener;
    dump.arrays{ctr+2} = px;
    dump.arrays{ctr+3} = py;
    dump.arrays{ctr+4} = pz;
    
    
    for b = 0:4
        ignoreit = GPU_setslab(DH, b, dump.arrays{ctr});
        ctr = ctr + 1;
    end
end

end
