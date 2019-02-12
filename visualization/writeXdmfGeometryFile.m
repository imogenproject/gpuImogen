function writeXdmfGeometryFile(fstring, geomgr)
% function writeXdmfGeometryFile(fstring, geomgr)
%>>fstring: filename to write to
%>>geomgr: GeometryManager class to fetch coordinates from

[x, y, z] = geomgr.ndgridSetIJK('pos', 'square');

q = single([x(:) y(:) z(:)]');

h5create(fstring, '/geometry_mesh', size(q), 'Datatype','single');

h5write(fstring, '/geometry_mesh', q);

end