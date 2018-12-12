function writeEnsightDatafiles(basename, frameNo, frame, varset)
% exportEnsightDatafiles(basename, frame #, data frame, {'names','of','vars'})
% basename: output filename base
% frame #:  frame number
% frame: Imogen sx_... savefile structure

twofluid = isfield(frame, 'mass2');

for n = 1:numel(varset)
    q = util_DerivedQty(frame, varset{n}, 0);
    if isa(q, 'struct') % var was a vector
	makeEnsightVectorFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q.X, q.Y, q.Z, varset{n});
    else
	makeEnsightScalarFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q, varset{n});
    end

    if twofluid && (strcmpi(varset{n}, '2fluid_dv') == 0)
        q = util_DerivedQty(frame, varset{n}, 1);
        if isa(q, 'struct') % var was a vector
            makeEnsightVectorFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q.X, q.Y, q.Z, varset{n});
        else
            makeEnsightScalarFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q, varset{n});
        end
    end
end

end
