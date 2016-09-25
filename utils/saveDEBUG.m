function saveDEBUG(data, namestring)

persistent savect;
if isempty(savect); savect = 0; end

mpidata = mpi_basicinfo();

if mpidata(2) == 0
  fprintf('Values for %s saved as dbsave # %i\n', namestring, savect);
end

fname = sprintf('dbsave_rank%i_%i.mat',mpidata(2),savect);
save(fname, 'data', 'namestring');

savect = savect + 1;

end

