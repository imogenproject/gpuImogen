function makeEnsightGeometryFile(SP, frameref, filename, reverseIndexOrder)
% This describes what the geometry file must output
% This must be followed TO THE LETTER and that INCLUDES SPACES
% See page 10 of Ensight Gold format PDF
GEOM = fopen([filename '.geom'], 'w');

% header stuff. six lines of exactly 80 chars each.
charstr = char(32*ones([400 1]));
charstr(1:8)     = 'C Binary';
charstr(81:93)   = 'Imogen export';
charstr(161:181) = 'Saved in Ensight Gold';
charstr(241:251) = 'node id off';
charstr(321:334) = 'element id off';
%charstr(401:407) = 'extents';
fwrite(GEOM, charstr, 'char*1');

% six floats, [xmin xmax ymin ymax zmin zmax]
extents = [0 0 0 0 0 0];

isUniform = 1;

if numel(frameref.dGrid{1}) == 1 % uniform spacing x
    extents(2) = frameref.dGrid{1} * max((size(frameref.mass, 1)-1), 1);
else
    extents(2) = sum(frameref.dGrid{1}(:,1,1));
    isUniform  = 0;
end

if numel(frameref.dGrid{2}) == 1 % uniform spacing y
    extents(4) = frameref.dGrid{2} * max((size(frameref.mass, 2)-1), 1);
else
    extents(4) = sum(frameref.dGrid{2}(1,:,1));
    isUniform  = 0;
end

if numel(frameref.dGrid{3}) == 1 % uniform spacing z
    extents(6) = frameref.dGrid{3} * max((size(frameref.mass, 3)-1), 1);
else
    extents(6) = sum(frameref.dGrid{3}(1,1,:));
    isUniform  = 0;
end

d = SP.returnInitializer();

if d.ini.geometry.pGeometryType == 2
    isUniform = 0;
    isCylindrical = 1;
else
    isCylindrical = 0;
end

%fwrite(GEOM, single(extents), 'float');

% part - exactly 80 chars
charstr = char(32*ones([80 1]));
charstr(1:4) = 'part';
fwrite(GEOM, charstr, 'char*1');

% NO Number of nodes - 1 int
%nnodes = prod(size(frameref.mass));
fwrite(GEOM, 1, 'int');

if isUniform % Easy-peasy
    % description line - exactly 80 chars
    % uniform block - exactly 80 c
    charstr = char(32*ones([160 1]));
    charstr(1:17) = 'simulation domain';
    charstr(81:93) = 'block uniform';
    fwrite(GEOM, charstr, 'char*1');

    % Write number of i j and k steps. 3 ints.
    domsize = size(frameref.mass);
    if numel(domsize) == 2; domsize(3) = 1; end
    
    orig_pos = [0 0 0 frameref.dGrid{1} frameref.dGrid{2} frameref.dGrid{3}];
    if reverseIndexOrder
        fwrite(GEOM, domsize([3 2 1]), 'int');
        fwrite(GEOM, orig_pos([3 2 1 6 5 4]), 'float');
    else
        fwrite(GEOM, domsize, 'int');
        fwrite(GEOM, orig_pos, 'float');
    end
elseif isCylindrical % must use curvilinear coordinates!!
    % description line - exactly 80 chars
    % uniform block - exactly 80 c
    charstr = char(32*ones([160 1]));
    charstr(1:17) = 'simulation domain';
    charstr(81:97) = 'block curvilinear';
    fwrite(GEOM, charstr, 'char*1');

    % Write number of i j and k steps. 3 ints.
    domsize = size(frameref.mass);
    if numel(domsize) == 2; domsize(3) = 1; end
    if reverseIndexOrder
        fwrite(GEOM, domsize([3 2 1]), 'int');
    else
        fwrite(GEOM, domsize, 'int');
    end

    % This turd-tastic hack recreates the original geometry manager, for one node, containing the global geometry info
    g = GeometryManager(d.ini.geometry.globalDomainRez); % fake global geometry manager
    switch d.ini.geometry.pGeometryType
    case ENUM.GEOMETRY_SQUARE
        g.geometrySquare(d.ini.geometry.affine, d.ini.geometry.d3h);
    case ENUM.GEOMETRY_CYLINDRICAL
        g.geometryCylindrical(d.ini.geometry.affine(1), round(2*pi/(d.ini.geometry.d3h(2)*d.ini.geometry.globalDomainRez(2))), d.ini.geometry.d3h(1), d.ini.geometry.affine(2), d.ini.geometry.d3h(3));
    end
    
    % fetch positions of all coordinates
    [xmat, ymat, zmat] = g.ndgridSetIJK('pos', 'square');

    if reverseIndexOrder
        fwrite(GEOM, permute(xmat, [3 2 1]), 'float');
        fwrite(GEOM, permute(ymat, [3 2 1]), 'float');
        fwrite(GEOM, permute(zmat, [3 2 1]), 'float');
    else
        fwrite(GEOM, xmat, 'float');
        fwrite(GEOM, ymat, 'float');
        fwrite(GEOM, zmat, 'float');
    end
else
    % description line - exactly 80 chars
    % uniform block - exactly 80 c
    charstr = char(32*ones([160 1]));
    charstr(1:17) = 'simulation domain';
    charstr(81:97) = 'block rectilinear';
    fwrite(GEOM, charstr, 'char*1');

    % Write number of i j and k steps. 3 ints.
    domsize = size(frameref.mass);
    if numel(domsize) == 2; domsize(3) = 1; end
    if reverseIndexOrder
        fwrite(GEOM, domsize([3 2 1]), 'int');
    else
        fwrite(GEOM, domsize, 'int');
    end

    ivec = cumsum(squish(frameref.dGrid{1}(:,1,1))) - frameref.dGrid{1}(1,1,1);
    jvec = cumsum(squish(frameref.dGrid{2}(1,:,1))) - frameref.dGrid{2}(1,1,1);
    kvec = cumsum(squish(frameref.dGrid{3}(1,1,:))) - frameref.dGrid{3}(1,1,1);

    if numel(ivec) ~= size(frameref.mass,1)
        ivec = (1:size(frameref.mass, 1))*frameref.dGrid{1}(1);
    end
    if numel(jvec) ~= size(frameref.mass,2)
        jvec = (1:size(frameref.mass, 2))*frameref.dGrid{2}(1);
    end
    if numel(kvec) ~= size(frameref.mass,3)
        kvec = (1:size(frameref.mass, 3))*frameref.dGrid{3}(1);
    end

    if reverseIndexOrder
        fwrite(GEOM, kvec, 'float');
        fwrite(GEOM, jvec, 'float');
        fwrite(GEOM, ivec, 'float');
    else
        fwrite(GEOM, ivec, 'float');
        fwrite(GEOM, jvec, 'float');
        fwrite(GEOM, kvec, 'float');
    end

end

fclose(GEOM);

end
