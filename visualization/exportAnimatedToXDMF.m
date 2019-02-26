function exportAnimatedToXDMF(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder)
% exportAnimatedToEnsight(SP, outBasename, inBasename, range, varset, timeNormalization)
%>> SP: SavefilePortal to access data from
%>> outBasename: Base filename for output Ensight files, e.g. 'mysimulation'
%>> inBasename:  Input filename for Imogen .mat savefiles, e.g. '2D_XY'
%>> range:       Set of savefiles to export (e.g. 0:50:1000)
%>> varset:      {'names','of','variables'} to save (see util_DerivedQty for list)
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units

if reverseIndexOrder
    warning('WARNING: reverseIndexOrder requested but exportAnimatedToXDMF does not support this. All output will retain original XYZ X-linear-stride order.');
end

autodetectVars = 0;
if strcmp(varset{1}, 'detect')
   disp('NOTICE: exportAnimatedToXDMF will detect all variables in first frame and export all.');
   autodetectVars = 1;
end



pertonly = 0;%input('Export perturbed quantities (1) or full (0)? ');
equilframe = [];

minf = mpi_basicinfo();
nworkers = minf(1); myworker = minf(2);
ntotal = numel(range); % number of frames to write
nstep = nworkers;

tic;

stepnums = zeros([ntotal 1]);

% part A - use ranks?

pg = ParallelGlobals();

fakeranks = 1;
c = pg.context; c.size = fakeranks;

for N = 1:fakeranks
    c.rank = N-1;
    pg.setNewContext(c);
    
    pg.setNewTopology(3, 'fakeAsHell');
    

    %fixme FIXME Fixme - problem, this is being acquired in various places as needed. Yuck. standardize
    %that process so we get it in one location.
    
    % Acquire geometry data in order to emit geometry file
    d = SP.getInitialConditions();
    g = GeometryManager(d.ini.geometry.globalDomainRez);
    switch d.ini.geometry.pGeometryType
        case ENUM.GEOMETRY_SQUARE
            g.geometrySquare(d.ini.geometry.affine, d.ini.geometry.d3h);
        case ENUM.GEOMETRY_CYLINDRICAL
            g.geometryCylindrical(d.ini.geometry.affine(1), round(2*pi/(d.ini.geometry.d3h(2)*d.ini.geometry.globalDomainRez(2))), d.ini.geometry.d3h(1), d.ini.geometry.affine(2), d.ini.geometry.d3h(3));
    end
    
    outgeo = [outBasename '_geometry%i.h5'];
    outgeo = sprintf(outgeo, N)
    disp('Emitting .h5 format geometry file...');
    writeXdmfGeometryFile(outgeo, g);
    disp('Success.');

end


outx = fopen([outBasename '_meta.xdmf'],'w');

% Begin emitting XDMF
% This could be written into a Matlab XML object but... this is simpler

% The prelude...
fprintf(outx, '<?xml version="1.0" ?>\n');
fprintf(outx, '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n');
fprintf(outx, '<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n');
% We have one domain (the simulation)
fprintf(outx, '  <Domain>\n');
% We have one "outer" grid which is the time ordered collection
fprintf(outx, '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n');
% The first inner grid occupies a special place in our hearts,
fprintf(outx, '      <!-- Frame 1 - special - contains actual definition of geometry -->\n');
fprintf(outx, '      <Grid Name="thedomain" GridType="Uniform">\n');
rez = d.ini.geometry.globalDomainRez;
% Because we define the actual topological and geometric data.
% FIXME: this emits a 3d structured mesh, but not all simulations will actually require this
% FIXME: some may use 3DRectMesh or 3DCoRectMesh
fprintf(outx, '        <Topology name="thetopo" TopologyType="3DSMesh" Dimensions="%i %i %i"/>\n', rez(3), rez(2), rez(1));
fprintf(outx, '        <Geometry name="thegeo" GeometryType="XYZ">\n');
% This refers to the emission from writeXdmfGeometryFile above
% file:/geometry_mesh is nothing but the serialized coordinate triplets of every cell in XYZ order:
% q(i,j,k) == [x(i,j,k) y(i,j,k) z(i,j,k)]
% then 
% [q(0,0,0) q(1,0,0) ... q(nx-1,0,0) q(0,1,0) ... q(nx-1, ny-1, 0) q(0,0,1), ... q(nx-1, ny-1, nz-1) ]
fprintf(outx, '          <DataItem Dimensions="%li 3" NumberType="Float" Precision="4" Format="HDF">%s:/geometry_mesh</DataItem>\n', int64(prod(rez)), outgeo);
fprintf(outx, '        </Geometry>\n');


% First frame is 'special' because of the above, so do it by itself then close </grid>:
SP.setMetamode(1); % do not load data, only metadata: extremely fast

datameta = SP.setFrame(range(1));
frname = SP.getSegmentFilename();

if autodetectVars
    if datameta.twoFluids
        if strcmp(datameta.varFmt, 'primitive')
            varset = {'mass','velX','velY','velZ','eint', 'mass2', 'velX2', 'velY2', 'velZ2', 'eint2'};
        else
            varset = {'mass','momX','momY','momZ','ener', 'mass2', 'momX2', 'momY2', 'momZ2', 'ener2'};
        end
    else
        if strcmp(datameta.varFmt, 'primitive')
            varset = {'mass','velX','velY','velZ','eint'};
        else
            varset = {'mass','momX','momY','momZ','ener'};
        end
    end
end

writeFrameAtts(outx, frname, datameta, rez, varset, timeNormalization);
fprintf(outx, '      </Grid>\n');

% Loop over all frames in the range, emitting a new <grid>...</grid> for each:
for N = 2:numel(range)
    SP.setFrame(range(N));
    frname = SP.getSegmentFilename();
    datameta = SP.getMetadata();
    tau = sum(datameta.time.history);
    fprintf(outx, '      <!-- Frame %i of output -->\n', int32(N));
    fprintf(outx, '      <Grid GridType="Uniform">\n');
    fprintf(outx, '        <Topology Reference="/Xdmf/Domain/Grid/Grid/Topology[@name=''thetopo'']" />\n');
    fprintf(outx, '        <Geometry Reference="/Xdmf/Domain/Grid/Grid/Geometry[@name=''thegeo'']" />\n');
    
    writeFrameAtts(outx, frname, datameta, rez, varset, timeNormalization);
    %fprintf(outx, '        <Time Value="%f" />\n', tau);
    %fprintf(outx, '        <Attribute Name="mass" Active="1" AttributeType="Scalar" Center="Node">\n');
    %fprintf(outx, '          <DataItem Dimensions="%i %i %i" NumberType="Float" Precision="4" Format="HDF">%s:/fluid1/mass</DataItem>\n', rez(3), rez(2), rez(1), frname);
    %fprintf(outx, '        </Attribute>\n');
    fprintf(outx, '      </Grid>\n');
end
% Close tags...
fprintf(outx, '    </Grid>\n');
fprintf(outx, '  </Domain>\n');
fprintf(outx, '</Xdmf>\n');

% And drop the mic
fclose(outx);

end

function writeFrameAtts(outx, frfile, frmeta, rez, varnames, timefactor)

tau = sum(frmeta.time.history);
fprintf(outx, '        <Time Value="%f" />\n', tau / timefactor);

for p = 1:numel(varnames)
    fprintf(outx, '        <Attribute Name="%s" Active="1" AttributeType="Scalar" Center="Node">\n', varnames{p});
    fprintf(outx, '          <DataItem Dimensions="%i %i %i" NumberType="Float" Precision="4" Format="HDF">%s:/fluid1/%s</DataItem>\n', rez(3), rez(2), rez(1), frfile, varnames{p});
    fprintf(outx, '        </Attribute>\n');
end


end


function out = subtractEquil(in, eq)
out = in;

out.mass = in.mass - eq.mass;
out.ener = in.ener - eq.ener;

out.momX = in.momX - eq.momX;
out.momY = in.momY - eq.momY;
out.momZ = in.momZ - eq.momZ;

out.magX = in.magX - eq.magX;
out.magY = in.magY - eq.magY;
out.magZ = in.magZ - eq.magZ;

end
