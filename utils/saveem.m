if exist('mass', 'var')
    MASS = mass.array;
    MOMX = mom(1).array;
    MOMY = mom(2).array;
    MOMZ = mom(3).array;
    ENER = ener.array;
end
if exist('run', 'var')
    MASS = run.fluid(1).mass.array;
    ENER = run.fluid(1).ener.array;
    MOMX = run.fluid(1).mom(1).array;
    MOMY = run.fluid(1).mom(2).array;
    MOMZ = run.fluid(1).mom(3).array;
end

save('~/saveem_dbg.mat','MASS','MOMX','MOMY','MOMZ','ENER');

