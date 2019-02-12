function exportAnimatedToVTK(SP, outBasename, range, varset, timeNormalization, reverseIndexOrder)
% exportAnimatedToEnsight(SP, outBasename, inBasename, range, varset, timeNormalization)
%>> SP: SavefilePortal to access data from
%>> outBasename: Base filename for output Ensight files, e.g. 'mysimulation'
%>> inBasename:  Input filename for Imogen .mat savefiles, e.g. '2D_XY'
%>> range:       Set of savefiles to export (e.g. 0:50:1000)
%>> varset:      {'names','of','variables'} to save (see util_DerivedQty for list)
%>> timeNormalization: Allows Imogen timestep-time to be converted into characteristic time units
	
addpath('~/vtkwriter');

if reverseIndexOrder
    warning('WARNING: reverseIndexOrder requested but exportAnimatedToVTK does not support this. All output will retain normal XYZ X-linear-stride order.');
end

pertonly = 0;%input('Export perturbed quantities (1) or full (0)? ');
equilframe = [];

minf = mpi_basicinfo();
nworkers = minf(1); myworker = minf(2);
ntotal = numel(range); % number of frames to write
nstep = nworkers;

tic;

stepnums = zeros([ntotal 1]);

%fixme FIXME Fixme - problem, this is being acquired in various places as needed. Yuck. standardize
%that process so we get it in one location. 

% GET GEOMETRY DATA HERE
d = SP.getInitialConditions();
g = GeometryManager(d.ini.geometry.globalDomainRez);
switch d.ini.geometry.pGeometryType
    case ENUM.GEOMETRY_SQUARE
        g.geometrySquare(d.ini.geometry.affine, d.ini.geometry.d3h);
    case ENUM.GEOMETRY_CYLINDRICAL
        g.geometryCylindrical(d.ini.geometry.affine(1), round(2*pi/(d.ini.geometry.d3h(2)*d.ini.geometry.globalDomainRez(2))), d.ini.geometry.d3h(1), d.ini.geometry.affine(2), d.ini.geometry.d3h(3));
end
% We require all cell positions in cartesian coordinates
[xp, yp, zp] = g.ndgridSetIJK('pos','square'); %#ok<ASGLU>

% WRITE GEOMETRY STRING:
% vtkwrite(filename, 'structured_grid', geomX, geomY, geomZ
outcmdstr = sprintf('vtkwrite(''%s%%04i.vtk'', ''structured_grid'', xp, yp, zp', outBasename);

ITER = myworker+1;
dataframe = SP.setFrame(range(ITER)); 

framedat = {};
for vn = 1:numel(varset)
    framedat{vn} = util_DerivedQty(dataframe, varset{vn}, 0);
end

% Loop over output varset:
% append to the vtkwrite string
for vn = 1:numel(varset)
    if isfield(framedat{vn}, 'X') % if isVector...
        outcmdstr = sprintf('%s, ''vectors'', ''%s'', framedat{%i}.X, framedat{%i}.Y, framedat{%i}.Z', outcmdstr, varset{vn}, vn, vn, vn);
    else
        outcmdstr = sprintf('%s, ''scalars'', ''%s'', framedat{%i}', outcmdstr, varset{vn}, vn);
    end
end
outcmdstr = sprintf('%s, ''binary'');', outcmdstr);

%framedat = {};

%--- Loop over all frames ---%
for ITER = (myworker+1):nstep:ntotal
    if ITER ~= (myworker+1); dataframe = SP.setFrame(range(ITER)); end

% FIXME this fails in parallel horribly...
    stepnums(ITER) = sum(dataframe.time.history) / timeNormalization;

    finalstr = sprintf(outcmdstr, ITER);
    % for elements of varset,
    % fetch using util_DerivedQty into a cell array
    if ITER ~= (myworker +1)
        for vn = 1:numel(varset)
            framedat{vn} = util_DerivedQty(dataframe, varset{vn}, 0); %#ok<AGROW>
        end
    end
    
    eval(finalstr);
    
    fprintf('%i ',myworker);
end

fprintf('Rank %i finished in %g sec.\n', myworker, toc);

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
