classdef GeometryManager < handle
% Global Index Semantics: Translates global index requests to local ones for lightweight global array support
% x = GlobalIndexSemantics(context, topology): initialize
% TF = GlobalIndexSemantics('dummy'): Check if previous init'd
% x = GlobalIndexSemantics(): Retreive

    properties (Constant = true, Transient = true)
        haloAmt = 4;
        SCALAR = 1000;
        VECTOR = 1001;
    end

    properties (SetAccess = public, GetAccess = public, Transient = true)
        
    end % Public

    properties (SetAccess = public, GetAccess = public)
        context;
        topology;
        globalDomainRezPlusHalos; % The size of global input domain size + halo added on [e.g. 512 500]
        
        pLocalDomainOffset; % The offset of the local domain taking the lower left corner of the
                            % input domain (excluding halo).

        edgeInterior; % Marks if [left/right, x/y/z] side is interior (circular) or exterior (maybe not)
        
        frameRotationCenter;
        frameRotationOmega;
    end % Private

    properties (SetAccess = private, GetAccess = public)
        pGeometryType;
        globalDomainRez;   % The input global domain size, with no halo added [e.g. 500 500]
        localDomainRez;          % The size of the local domain + any added halo [256 500
        
        % The local parts of the x/y/z index counting vectors
        % Matlab format (from 1)
        localIcoords; localJcoords; localKcoords;
        
        affine; % Stores [dx dy dz] or [r0 z0]
        d3h; % The [dx dy dz] or [dr dphi dz] spacing
        pInnerRadius;   % Used only if geometryType = ENUM.GEOMETRY_CYLINDRICAL

        localXposition; % X coordinate, cell-centered, on this node (cartesian geometry)
        localYposition; % Y coordinate, cell-centered, on this node (cartesian geometry)
        localZposition; % Z coordinate, cell-centered, on this node (cartesian & cylindrical geometry)
        localRposition; % R coordinate, cell-centered, on this node (cylindrical geometry
        localPhiPosition; % Angular coordinate, cell centered, on this node (cylindrical geometry)
        
        % FIXME: Add spacing-vectors for calculating updates over nonuniform grids here
    end % Readonly

    properties (SetAccess = private, GetAccess = private)
        % Marks whether to wrap or continue counting at the exterior halo
        circularBCs;

        % Local indices such that validArray(nohaloXindex, ...) contains everything but the halo cells
        % Useful for array reducing operations
        nohaloXindex, nohaloYindex, nohaloZindex;
    end

    properties (Dependent = true)
        
    end % Dependent

    methods
        function obj = GeometryManager(globalResolution, circularity)
            % GlobalIndexSemantics sets up Imogen's global/local indexing
            % system. It automatically fetches itself the context & topology stored in ParallelGlobals
            
            parInfo      = ParallelGlobals();
            obj.context  = parInfo.context;
            obj.topology = parInfo.topology;
            
            % Accept a serialized self representation and shortcut to the deserializer
            if isa(globalResolution, 'struct')
                obj.deserialize(globalResolution);
                return;
            end
            
            if nargin < 1
                warning('GeometryManager received no resolution: going with 512x512x1; Call geo.setup([nx ny nz]) to change.');
                globalResolution = [512 512 1];
            end
            if nargin < 2
                circularity = [1 1 1];
            end

            obj.setup(globalResolution, circularity);
            obj.geometrySquare([0 0 0], [1 1 1]); % Set default geometry to cartesian, unit spacing
            
            obj.frameRotationCenter = [0 0 0];
            obj.frameRotationOmega = 0;
        end

        function package = serialize(self)
            % Converts the geo object into a structure which can reinitialize itself
            % 'context' and 'topology' props are excluded because they do not persist across invocations if restarting
            package = struct('globalDomainRezPlusHalos', self.globalDomainRezPlusHalos, ...
                             'localDomainRez', self.localDomainRez, 'pLocalDomainOffset',self.pLocalDomainOffset, ...
                             'globalDomainRez', self.globalDomainRez, 'edgeInterior', self.edgeInterior, ...
                             'pGeometryType', self.pGeometryType, 'pInnerRadius', self.pInnerRadius, ...
                             'localIcoords', self.localIcoords, 'localJcoords', self.localJcoords, ...
                             'localKcoords', self.localKcoords, 'circularBCs', self.circularBCs, ...
                             'nohaloXindex', self.nohaloXindex, 'nohaloYindex', self.nohaloYindex, ...
                             'nohaloZindex', self.nohaloZindex, 'affine', self.affine, 'd3h', self.d3h, ...
                             'localXposition', self.localXposition, 'localYposition', self.localYposition, ...
                             'localZposition', self.localZposition, 'localRposition', self.localRposition, ...
                             'localPhiPosition', self.localPhiPosition);
        end
        
        function deserialize(self, package)
            % geo.deserialize(package) accepts a package structure as returned by package = geo.serialize()
            % And overwrites this object's fields with the package's values verbatim.
            F = fields(package);
            for N = 1:numel(F)
                self.(F{N}) = package.(F{N});
            end
        end
        
        function setup(obj, global_size, iscirc)
            % geo.setup(global_resolution) establishes the global resolution &
            % runs the bookkeeping for it, determining what set of I/J/K vectors out of the global grid that
            % this node has.
            % geo.setup(global_resolution, Initializer) sets the resolution and
            if numel(global_size) < 3; global_size(3) = 1; end
            if numel(global_size) < 2; global_size(2) = 1; end

            % Default to circular BCS
            obj.circularBCs = [1 1 1];
            % If we receive a bcMode structure 
            if nargin == 3
                bcs = BCManager.expandBCStruct(iscirc);
                for i = 1:3
                    obj.circularBCs(i) = 1*(strcmp(bcs(1,i),'circ'));
                end
            end
            dblhalo = 2*obj.haloAmt;
            
            obj.globalDomainRez = global_size;
            obj.globalDomainRezPlusHalos   = ...
                global_size + ... % real cells 
                dblhalo*double(obj.topology.nproc-1).*double(obj.topology.nproc > 1) + ... % internal edge|edge halo
                dblhalo*obj.circularBCs.*double(obj.topology.nproc > 1); % external wraparound halo if circular BC
            
            % Default size: Ntotal / Nproc - all ranks will compute the
            % same value here
            propSize = floor(global_size ./ double(obj.topology.nproc));
            real_cells = propSize;
            
            % add possible left and right side halos
            propSize = propSize + obj.haloAmt*(obj.nodeHasHalo(0) + obj.nodeHasHalo(1));

            % Compute offset: just real cell count times offset
            obj.pLocalDomainOffset = real_cells .* obj.topology.coord;
            
            % If we're at the plus end of the topology in a dimension, increase proposed size to meet global domain resolution.
            % in event that Ncells / Nprocs != integer
            deficit = obj.globalDomainRez - obj.topology.nproc .* real_cells;
            for i = 1:obj.topology.ndim
                if (deficit(i) > 0) && (obj.topology.coord(i) == (obj.topology.nproc(i)-1))
                    propSize(i) = propSize(i) + deficit(i);
                end
            end

            obj.localDomainRez = propSize;

            obj.edgeInterior(1,:) = double(obj.topology.coord > 0);
            obj.edgeInterior(2,:) = double(obj.topology.coord < (obj.topology.nproc-1));

            obj.updateGridVecs();
        end % Constructor

        function yn = nodeHasHalo(obj, side)
            if side==0 % do we have a left halo? If > 1 processor & we are not first, or > 1 processor and circular then yes!
                yn = (obj.topology.nproc > 1).*((obj.topology.coord > 0) + obj.circularBCs);
                yn = 1*(yn > 0);
            else % do we have a right halo? If > 1 processor and (we are not last, or bc is circular) then yes!
                yn = (obj.topology.nproc > 1).*((obj.topology.coord < (obj.topology.nproc-1)) + obj.circularBCs);
                yn = 1*(yn > 0);
            end
        end
        
        function makeDimCircular(obj, dim)
            % makeDimCircular(1 <= dim <= 3) declares a circular BC on dim
            % Effect: Outer edge has a halo
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 1;
            obj.updateGridVecs();
        end

        function makeDimNotCircular(obj, dim)
            % makeDimNotCircular(1 <= dim <= 3) declares a noncircular BC on dim
            % Effect: Outer e:qdge does not have a halo.
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 0;
            obj.updateGridVecs();
        end

        function makeBoxSize(obj, newsize)
            % geo.makeBoxSize([s])
            % If geometry is square, geo.makeBoxSize([a b c]) sets grid spacing to [a/nx, b/ny, c/nz].
            % If given other than a 3-element vector, [nx ny nz]*s(1)/nx is used (i.e. a box of
            % length s(1) and with square cells)
            % If ny or nz = 1, the respective spacing is set to 1.
            %
            % If geometry is cylindrical,
            % geo.makeBoxSize([width angle height]) sets grid spacing such that the cylindrical annulus
            % spans (rin, rin+width) x (angle) x (0, height)
            % If other than 3 element vector is given, a = x(1) and spacing is set for 
            % width = height = a, and angle = 2*pi
            % Note that for cylinrical geometry,
            % is is very strongly advised that angle be 2*pi / M, for M in \mathbb{Z} > 0.
            
            if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                originCoord = -obj.affine ./ obj.d3h;
                
                if numel(newsize) ~= 3; newsize = obj.globalDomainRez * newsize(1) / obj.globalDomainRez(1); end
                obj.d3h = newsize ./ obj.globalDomainRez;
                if obj.globalDomainRez(3) == 1; obj.d3h(3) = 1; end
                if obj.globalDomainRez(2) == 1; obj.d3h(2) = 1; end
                
                % Restore the correct origin coordinate
                obj.makeBoxOriginCoord(originCoord);
            end
            if obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                if numel(newsize) ~= 3 % make the annulus (rin,rin+x) x (0,z=x) x (0,2pi)
                    width  = newsize(1);
                    height = newsize(1);
                    ang    = 2*pi;
                else
                    width  = newsize(1);
                    ang    = newsize(2);
                    height = newsize(3);
                end
                obj.d3h = [width ang height] ./ obj.globalDomainRez;
            end
            obj.updateGridVecs();
        end
        
        function makeBoxOriginCoord(obj, coord)
            % geometry.makeBoxOriginCoord(i [j [k]])
            % Sets the box's affine parameter such that cell center coordinate i
            % corresponds to an X position of zero.
            % The same is done for j and Y, and k and Z, if j and/or k are given.
            
            obj.makeBoxLLPosition(1-coord, 'incells');
        end
        
        function makeBoxLLPosition(obj, position, normalization)
            % geometry.makeBoxLLPosition(x [y [z]], ['incells'])
            % directly sets the geometry.affine parameter such that cell
            % (1,1,1) has this position. If given 'incells', position is
            % taken as being measured in cells.
            
            if nargin == 3
               if strcmp(normalization, 'incells') == 1
                  b = numel(position); position(1:b) = position(1:b) .* obj.d3h(1:b); 
               end
            end
            
            obj.affine(1) = position(1);
            if numel(position) > 1; obj.affine(2) = position(2); end
            if numel(position) > 2; obj.affine(3) = position(3); end
            
            obj.updateGridVecs();
        end

        function geometrySquare(obj, zeropos, spacing)
            % geometrySquare([x0 y0 z0], [hx hy hz])
            % sets the (1,1,1)-index cell to have center at [x0 y0 z0]
            % and grid spacing to [hx hy hz]
            % If no spacing is given, defaults to [1 1 1]
            % If no zero gauge is given, defaults to [0 0 0].
            obj.pGeometryType = ENUM.GEOMETRY_SQUARE;
            obj.pInnerRadius = 1.0; % ignored

            if isa(spacing, 'cell')
               spacing = [spacing{1} spacing{2} spacing{3}]; 
            end
            
            if nargin >= 2
                if numel(zeropos) ~= 3; zeropos = [1 1 1]*zeropos(1); end
            else
                zeropos = [0 0 0];
            end
            obj.affine = zeropos;   
            if nargin >= 3
                if numel(spacing) ~= 3; spacing = [1 1 1]*spacing(1); end
            else
                spacing = [1 1 1];
            end
            obj.d3h    = spacing;

            obj.updateGridVecs();
        end

        function geometryCylindrical(obj, Rin, M, dr, z0, dz)
            % geometryCylindrical(Rin, M, dr, z0, dz) sets the
            % annulus' innermost cell's center to Rin with spacing dr,
            % and sets dphi so we span 2*pi/M in angle.
            obj.pGeometryType = ENUM.GEOMETRY_CYLINDRICAL;
            obj.makeDimNotCircular(1); % updates grid vecs

           % This is used in the low-level routines as the Rcenter of the innermost cell of this node...
            obj.pInnerRadius  = Rin + dr*(obj.localIcoords(1)-1);

            obj.affine = [Rin z0];
            dphi = 2*pi / (M*obj.globalDomainRez(2));
            obj.d3h = [dr dphi dz];

            obj.updateGridVecs();
        end
        
        function [u, v, w] = toLocalIndices(obj, x, y, z)
            % [u, v, w] = geo.toLocalIndices(x, y, z) converts a global set of coordinates to 
            %     local coordinates, and keeps only those in the local domain
	    %     If no elements are, the corresponding set is []
	    %     This mode acts independently on x, y, and z, i.e. does not form any outer products
	    %     [u, ~, ~] = geo.toLocalIndices( [x(:) y(:) z(:)] ) converts the tuples to local indices
	    %     and again keeps only those which lie inside this node's domain and returns them in u
            u = []; v = []; w = [];
            if (nargin == 2) && (size(x,2) == 3)
                z = x(:,3) - obj.pLocalDomainOffset(3);
                y = x(:,2) - obj.pLocalDomainOffset(2);
                x = x(:,1) - obj.pLocalDomainOffset(1);
  
                keep = (x>0) & (x<=obj.localDomainRez(1)) & (y>0) & (y<=obj.localDomainRez(2)) & (z>0) & (z<=obj.localDomainRez(3));
                u = [x(keep) y(keep) z(keep)];
                return;
            end
            if (nargin >= 2) && (~isempty(x)); x = x - obj.pLocalDomainOffset(1); u=x((x>0)&(x<=obj.localDomainRez(1))); end
            if (nargin >= 3) && (~isempty(y)); y = y - obj.pLocalDomainOffset(2); v=y((y>0)&(y<=obj.localDomainRez(2))); end
            if (nargin >= 4) && (~isempty(z)); z = z - obj.pLocalDomainOffset(3); w=z((z>0)&(z<=obj.localDomainRez(3))); end

        end

        function [x, y, z] = toCoordinates(obj, I0, Ix, Iy, Iz, h, x0)
        % [x y z] = toCoordinates(obj, I0, Ix, Iy, Iz, h, x0) returns (for n = {x, y, z})
        % (In - I0(n))*h(n) - x0(n), i.e.
        % I0 is an index offset, h the coordinate spacing and x0 the coordinate offset.
        % [] for I0 and x0 default to zero; [] for h defaults to 1; scalars are multiplied by [1 1 1]
        % Especially useful during simulation setup, converting cell indices to physical positions
        % to evaluate functions at.

        if nargin < 3; error('Must receive at least toCoordinates(I0, x)'); end
        % Throw duct tape at the arguments until glaring deficiencies are covered
        if numel(I0) ~= 3; I0 = [1 1 1]*I0(1); end
        if nargin < 4; Iy = []; end
        if nargin < 5; Iz = []; end
        if nargin < 6; h = [1 1 1]; end
        if numel(h) ~= 3; h = [1 1 1]*h(1); end
        if nargin < 7; x0 = [1 1 1]; end
        if numel(x0) ~= 3; x0 = [1 1 1]*x0(1); end

        if ~isempty(Ix); x = (Ix - I0(1))*h(1) - x0(1); end
        if ~isempty(Iy); y = (Iy - I0(2))*h(2) - x0(2); end
        if ~isempty(Iz); z = (Iz - I0(3))*h(3) - x0(3); end

        end

        function Y = evaluateFunctionOnGrid(obj, afunc)
        % If geometry is set to square,
        % F = evaluateFunctionOnGrid(@func) returns func(x, y, z) using the
        % x/y/z positions of every cell center
        % If geometry is set to cylindrical,
        % F = evaluateFunctionOnGrid(@func) returns func(r, phi, z) using the
        % r/phi/z positions of every cell center.
            [x, y, z] = obj.ndgridSetIJK('pos');
            Y = afunc(x, y, z);
        end
        
        function localset = LocalIndexSet(obj, globalset, d)
        % Extracts the portion of ndgrid(1:globalsize(1), ...) visible to this node
        % Renders the 3 edge cells into halo automatically
        
            pLocalMax = obj.pLocalDomainOffset + obj.localDomainRez;

            if nargin == 3
                q = globalset{d};
                localset =  q((q >= obj.pLocalDomainOffset(d)) & (q <= pLocalMax(d))) - obj.pLocalDomainOffset(d) + 1;
            else
                localset = cell(3,1);
                
                for n = 1:min(numel(obj.globalDomainRezPlusHalos), numel(globalset))
                    q = globalset{n};
                    localset{n} = q((q >= obj.pLocalDomainOffset(n)) & (q <= pLocalMax(n))) - obj.pLocalDomainOffset(n) + 1;
                end
            end
        end
        
        function LL = cornerIndices(obj)
        % Return the index of the lower left corner in both local and global coordinates
        % Such that subtracting them from a 1:size(array) will make the lower-left corner that isn't part of the halo [0 0 0] in local coords and whatever it is in global coords
        ndim = numel(obj.globalDomainRezPlusHalos);

            LL=[1 1 1; (obj.pLocalDomainOffset+1)];
            for j = 1:ndim
                LL(j,:) = LL(j,:) - obj.haloAmt*(obj.topology.nproc(j) > 1);
            end

        end

        function updateGridVecs(obj)
            % geo.updateGridVecs(). Utility - upon change in global dims, recomputes x/y/z index
            % vectors
            ndim = numel(obj.globalDomainRezPlusHalos);

            x   = cell(ndim,1);
            lnh = cell(ndim,1);
            lefthalo = obj.nodeHasHalo(0);
            
            for j = 1:ndim
                q = 1:obj.localDomainRez(j);
                % This line degerates to the identity operation if nproc(j) = 1
                q = q + obj.pLocalDomainOffset(j) - obj.haloAmt*lefthalo(j);

                % If the edges are periodic, wrap coordinates around
                if (obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1)
                    q = mod(q + obj.globalDomainRez(j) - 1, obj.globalDomainRez(j)) + 1;
                end
                x{j} = q;

                lmin = 1; lmax = obj.localDomainRez(j);
                if (obj.topology.coord(j) > 0) || ((obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1))
                    lmin = 1 + obj.haloAmt; % If the minus side is a halo
                end
                if (obj.topology.coord(j) < obj.topology.nproc(j)-1) || ((obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1))
                    lmax = lmax - obj.haloAmt; % If the plus side is a halo
                end

                lnh{j} = lmin:lmax;
            end
            
            if ndim == 2; x{3} = 1; end
            
            obj.localIcoords = x{1}; obj.localJcoords = x{2}; obj.localKcoords = x{3};
            obj.nohaloXindex = lnh{1}; obj.nohaloYindex = lnh{2}; obj.nohaloZindex = lnh{3};
            
            if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                obj.localXposition = obj.affine(1) + obj.d3h(1) * (obj.localIcoords-1);
                obj.localYposition = obj.affine(2) + obj.d3h(2) * (obj.localJcoords-1);
                obj.localZposition = obj.affine(3) + obj.d3h(3) * (obj.localKcoords-1);
            end
            
            if obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                obj.localRposition = obj.affine(1) + obj.d3h(1)*(obj.localIcoords-1);
                obj.localPhiPosition = obj.d3h(2)*(obj.localJcoords-1);
                obj.localZposition = obj.affine(2) + obj.d3h(3)*(obj.localKcoords-1);
                
            end
        end

        function [u, v, w] = ndgridVecs(obj, form)
            % [u v w] = geo.ndgridVecs(form) returns the x-, y- and z- index vectors
            % if 'form' is absent, returns the integer grid indices
            % if 'form' is the string 'pos', return 

            if nargin < 2; form = 'coords'; end
            
            if strcmp(form,'coords')
                u = obj.localIcoords; v = obj.localJcoords; w = obj.localKcoords;
            end
            if strcmp(form, 'pos')
                if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                    u = obj.localXposition;
                    v = obj.localYposition;
                    w = obj.localZposition;
                end
                if obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                    u = obj.localRposition;
                    v = obj.localPhiPosition;
                    w = obj.localZposition;
                end
                
            end
        end

        function [x, y, z] = ndgridSetIJK(obj, form, geotype)
            % [x y z] = ndgridsetIJK(['pos' | 'coords'], ['square','cyl']) returns the part of
            % the global domain that lives on this node.
            % If the 1st argument is 'pos' returns the ndgrid() of the "physical" positions that the code
            % will use (see setBoxSize/setBoxOriginCoord/setBoxLLPosition/geometryCylindrical)
            % If 'coords', returns the ndgrid() of the cell coordinates (numbered from 1) instead.
            % If no argument, defaults to 'coords'.
            % If computing positions & the 2nd argument is 'square', the returned positions
            % will be in cartesian form (cylindrical will be converted)
            % If computing positions & the 2nd argument is 'cyl', the returned position
            % will be in polar form (square will be converted).
            
            if nargin < 2; form = 'coords'; end
            
            if strcmp(form, 'coords')
                [x, y, z] = ndgrid(obj.localIcoords, obj.localJcoords, obj.localKcoords);
                return;
            end
            if strcmp(form, 'pos')
                if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                    [x, y, z] = ndgrid(obj.localXposition, obj.localYposition, obj.localZposition);
                    if (nargin == 3) && strcmp(geotype, 'cyl') % cvt output to cylindrical
                        a = x; b = y;
                        x = sqrt(a.^2+b.^2);
                        y = atan2(b,a);
                        y = y + 2*pi*(y<0); % go from 0 to 2pi, not -pi to pi
                    end
                elseif obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                    [x, y, z] = ndgrid(obj.localRposition, obj.localPhiPosition, obj.localZposition);
                    if (nargin == 3) && strcmp(geotype, 'square') % cvt output to square
                        r = x; phi = y;
                        x = r.*cos(phi);
                        y = r.*sin(phi);
                    end
                else
                    error('FATAL: Geometry type is invalid!!!');
                end
                return;
            end
            error('ndgridSetIJK called but form was %s, not ''coords'', or ''pos''.\n', form);
        end
        
        function [x, y] = ndgridSetIJ(obj, form)
            % [x, y] = ndgridsetIJ(['pos' | 'coords'], ['square' | 'cyl']) returns the part of
            % the X-Y or R-Phi global domain that lives on this node.
            % If the argument is 'pos' returns the ndgrid() of the "physical" positions that the code
            % will use (see setBoxSize/setBoxOriginCoord/setBoxLLPosition/geometryCylindrical)
            % If 'coords', returns the ndgrid() of the cell coordinates (numbered from 1) instead.
           % If 'pos' is followed by 'square' or 'cyl', output will be in XY or R-Theta coordinates
           % even if the geometry is cylindrical or square, respectively.
            % If no argument, defaults to 'coords'.
            
            if nargin < 2; form = 'coords'; end
            
            if strcmp(form, 'coords')
                [x, y] = ndgrid(obj.localIcoords, obj.localJcoords);
                return;
            end
            if strcmp(form, 'pos')
                if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                    [x, y] = ndgrid(obj.localXposition, obj.localYposition);
                  if (nargin == 3) && strcmp(geotype, 'cyl') % convert xy to r-theta
                     a = x; b = y;
                     x = sqrt(a.^2+b.^2);
                     y = 2*pi*(y<0) + atan2(b,a);
                  end
                elseif obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                    [x, y] = ndgrid(obj.localRposition, obj.localPhiPosition);
                  if (nargin == 3) && strcmp(geotype,'square')
                      r = x; phi = y;
                      x = r .* cos(phi);
                      y = r .* sin(phi);
                  end
                end
                return;
            end
            error('ndgridSetIJ called but form was %s, not ''coords'', or ''pos''.\n', form);
        end
        
        function [y, z] = ndgridSetJK(obj, form)
            % [y, z] = ndgridsetJK(['pos' | 'coords']) returns the part of
            % the Y-Z or Phi-Z global domain that lives on this node.
            % If the argument is 'pos' returns the ndgrid() of the "physical" positions that the code
            % will use (see setBoxSize/setBoxOriginCoord/setBoxLLPosition/geometryCylindrical)
            % If 'coords', returns the ndgrid() of the cell coordinates (numbered from 1) instead.
            % If no argument, defaults to 'coords'.
            
            if nargin < 2; form = 'coords'; end
            
            if strcmp(form, 'coords')
                [y, z] = ndgrid(obj.localJcoords, obj.localKcoords);
                return;
            end
            if strcmp(form, 'pos')
                if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                    [y, z] = ndgrid(obj.localYposition, obj.localZposition);
                elseif obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                    [y, z] = ndgrid(obj.localPhiPosition, obj.localZposition);
                end
                return;
            end
            
            error('ndgridSetJK called but input argument was %s, not ''coords'', or ''pos''.\n', form);
        end
        
        function [x, z] = ndgridSetIK(obj, form)
            % [x, z] = ndgridsetIK(['pos' | 'coords']) returns the part of
            % the X-Z or R-Z global domain that lives on this node.
            % If the argument is 'pos' returns the ndgrid() of the "physical" positions that the code
            % will use (see setBoxSize/setBoxOriginCoord/setBoxLLPosition/geometryCylindrical)
            % If 'coords', returns the ndgrid() of the cell coordinates (numbered from 1) instead.
            % If no argument, defaults to 'coords'.
            if nargin < 2; form = 'coords'; end
            
            if strcmp(form, 'coords')
                [x, z] = ndgrid(obj.localIcoords, obj.localKcoords);
                return;
            end
            if strcmp(form, 'pos')
                if obj.pGeometryType == ENUM.GEOMETRY_SQUARE
                    [x, z] = ndgrid(obj.localXposition, obj.localZposition);
                elseif obj.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
                    [x, z] = ndgrid(obj.localRposition, obj.localZposition);
                end
                return;
            end
            
            error('ndgridSetIK called but input argument was %s, not ''coords'', or ''pos''.\n', form);
        end

       function makesize = localDimsFor(obj, dimtype)

            switch dimtype
                case 1; makesize = [obj.localDomainRez(1) 1 1];
                case 2; makesize = [1 obj.localDomainRez(2) 1];
                case 3; makesize = [1 1 obj.localDomainRez(3)];
                case 4; makesize = [obj.localDomainRez(1:2) 1];
                case 5; makesize = [obj.localDomainRez(1) 1 obj.localDomainRez(3)];
                case 6; makesize = [1 obj.localDomainRez(2:3)];
                case 7; makesize = obj.localDomainRez;
            end

        end

        % Generic function that the functions below talk to
        function out = makeValueArray(obj, dims, dtype, val)
            makesize = obj.localDimsFor(dims);
            
            % Slap a 3 on the first dim to build a vector
            % This is stupid and REALLY should be exchanged (3 goes LAST)
            if (nargin > 2) && (dtype == obj.VECTOR); makesize = [3 makesize]; end

            % Generate an array of 0 if not given a value
            if (nargin < 4); val = 0; end

            if isnan(val); out = rand(makesize); else; out = val*ones(makesize); end
        end 

        
        function O = zerosXY(obj, dtype)
        % These generate a set of zeros of the size of the part of the global grid residing on this node    
            if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(4, dtype, 0); end
        function O = zerosXZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(5, dtype, 0); end
        function O = zerosYZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(6, dtype, 0); end
        function O = zerosXYZ(obj, dtype); if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(7, dtype, 0); end

        
        function O = onesXY(obj, dtype)
            % These generate a set of ones of the size of the part of the global grid residing on this node
            if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(4, dtype, 1); end
        function O = onesXZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(5, dtype, 1); end
        function O = onesYZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(6, dtype, 1); end
        function O = onesXYZ(obj, dtype); if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(7, dtype, 1); end

        function O = randsXY(obj, dtype)
            % These generate a set of random #s of the size of the part of the global grid residing on this node
            if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(4, dtype, NaN); end
        function O = randsXZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(5, dtype, NaN); end
        function O = randsYZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(6, dtype, NaN); end
        function O = randsXYZ(obj, dtype); if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(7, dtype, NaN); end


        function [rho, mom, mag, ener] = basicFluidXYZ(obj)
            % Returns a single fluid [rho, mom, mag, ener] with
            % rho = 1, mom = [0 0 0], B = [0 0 0] and ener = 1 (pressure = gamma-1)
            rho = ones(obj.localDomainRez);
            mom = zeros([3 obj.localDomainRez]);
            mag = zeros([3 obj.localDomainRez]);
            ener= ones(obj.localDomainRez);
        end
        
        function G = getNodeGeometry(obj)
            % Function returns an nx X ny X nz array whose (i,j,k) index contains the rank of the
            % node residing on the nx X ny X nz topology at that location.
            G = zeros(obj.topology.nproc);
            xi = mpi_allgather( obj.topology.coord(1) );
            yi = mpi_allgather( obj.topology.coord(2) );
            zi = mpi_allgather( obj.topology.coord(3) );
            i0 = xi + obj.topology.nproc(1)*(yi + obj.topology.nproc(2)*zi) + 1;
            G(i0) = 0:(numel(G)-1);
        end

        % This is the only function in geo that references any part of the topology execpt .nproc or .coord
        function index = getMyNodeIndex(obj)
            index = double(mod(obj.topology.neighbor_left+1, obj.topology.nproc));
        end

        function slim = withoutHalo(obj, array)
            % slim = geo.withoutHalo(fat), when passed an array of size equal to the node's
            % array size, returns the array with halos removed.

            cantdo = 0;
            for n = 1:3; cantdo = cantdo || (size(array,n) ~= obj.localDomainRez(n)); end
                
            cantdo = mpi_max(cantdo); % All nodes must receive an array of the appropriate size
  
            if cantdo
                disp(obj.topology);
                disp([size(array); obj.localDomainRez]);
                error('Oh teh noez, rank %i received an array of invalid size!', mpi_myrank());
            end
            slim = array(obj.nohaloXindex, obj.nohaloYindex, obj.nohaloZindex);
        end
        
        function DEBUG_setTopoSize(obj, n)
            % This function is only for debugging: It allows the topology's .nproc parameter to
            % be overwritten
            obj.topology.nproc = n;

            c = obj.circularBCs;
            obj.setup(obj.globalDomainRez);
            for x = 1:n; if c(x) == 0; obj.makeDimNotCircular(x); end; end
        end
        function DEBUG_setTopoCoord(obj,c)
            % This fucntion is only for debugging: It allows the topology's .coord parameter to
            % be overwritten
            obj.topology.coord = c;
            c = obj.circularBCs;
            obj.setup(obj.globalDomainRez);
            for x = 1:3; if c(x) == 0; obj.makeDimNotCircular(x); end; end
        end

    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

