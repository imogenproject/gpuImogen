function writeEnsightDatafiles(basename, frameNo, frame, varset, reverseIndexOrder)
% exportEnsightDatafiles(basename, frame #, data frame, {'names','of','vars'}, reverseIndexOrder)
% basename: output filename base
% frame #:  frame number
% frame: Imogen sx_... savefile structure
% varset: {'names', 'of', 'util_DerivedQty', 'fields'}
% reverseIndexOrder: if true, [XYZ] written in [ZYX] order (off if not present)

twofluid = isfield(frame, 'mass2');
if nargin < 5; reverseIndexOrder = 0; end

for n = 1:numel(varset)
    q = util_DerivedQty(frame, varset{n}, 0);
    if isa(q, 'struct') % var was a vector
        makeEnsightVectorFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q.X, q.Y, q.Z, varset{n}, reverseIndexOrder);
    else
        makeEnsightScalarFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q, varset{n}, reverseIndexOrder);
    end
    
    if twofluid && (strcmpi(varset{n}, '2fluid_dv') == 1)
        q = util_DerivedQty(frame, varset{n}, 1);
        if isa(q, 'struct') % var was a vector
            makeEnsightVectorFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q.X, q.Y, q.Z, varset{n}, reverseIndexOrder);
        else
            makeEnsightScalarFile(sprintf('%s.%s.%04i', basename, varset{n}, frameNo), q, varset{n}, reverseIndexOrder);
        end
    end
end

end
