function saveDEBUG(data, namestring)

persistent savect;
if isempty(savect); savect = 0; end
GIS = GlobalIndexSemantics();

% gather data onto rank 0


if GIS.context.rank == 0
  fprintf('Values for %s saved as dbsave # %i\n', namestring, savect);
end

  fname = sprintf('dbsave_rank%i_%i.mat',GIS.context.rank,savect);
  save(fname, 'data', 'namestring');
%end

savect = savect + 1;


end
