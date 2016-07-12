classdef GlobalIndexSemantics < handle
% Global Index Semantics: Translates global index requests to local ones for lightweight global array support
% x = GlobalIndexSemantics(context, topology): initialize
% TF = GlobalIndexSemantics('dummy'): Check if previous init'd
% x = GlobalIndexSemantics(): Retreive

    properties (Constant = true, Transient = true)
        haloAmt = 3;
        SCALAR = 1000;
        VECTOR = 1001;
    end

    properties (SetAccess = public, GetAccess = public, Transient = true)
    end % Public

    properties (SetAccess = public, GetAccess = public)
        context;
        topology;
        pGlobalDomainRezPlusHalos; % The size of global input domain size + halo added on [e.g. 512 500]
        pLocalRez;     % The size of the local domain + any added halo [256 500
        pLocalDomainOffset;   % The offset of the local domain taking the lower left corner of the
                     % input domain (excluding halo).
        pGlobalDomainRez;   % The input global domain size, with no halo added [e.g. 500 500]

        edgeInterior; % Whether my [ left or right, x/y/z ] side is interior & therefore circular, or exterior
                      %                                          The Oxford comma, it matters!! -^
        
    end % Private

    properties (SetAccess = private, GetAccess = public)
        pGeometryType;
        pInnerRadius;   % Used only if geometryType = ENUM.GEOMETRY_CYLINDRICAL
    end % Readonly

    properties (SetAccess = private, GetAccess = private)
        % The local parts of the x/y/z index counting vectors
        % Matlab format (from 1)
        localXvector; localYvector; localZvector;

        % Marks whether to wrap or continue counting at the exterior halo
        circularBCs;

        % Local indices such that validArray(nohaloXindex, ...) contains everything but the halo cells
        % Useful for array reducing operations
        nohaloXindex, nohaloYindex, nohaloZindex;

        localXposition; localYposition; localZposition;
    end

    properties (Dependent = true)
    end % Dependent

    methods
        function obj = GlobalIndexSemantics(context, topology)
            % GlobalIndexSemantics sets up Imogen's global/local indexing
            % system. Requires a context & topology from PGW's
            % parallel_start; Will fudge serial operation if it does not
            % get them.
            persistent instance;

            if ~(isempty(instance) || ~isvalid(instance))
                obj = instance; return;
            end

            if nargin < 2;
                warning('GlobalIndexSemantics received no topology: generating one');
                if nargin < 1;
                    warning('GlobalIndexSemantics received no context: generating one.');
                    obj.context = parallel_start();
                end
                obj.topology = parallel_topology(obj.context, 3);
            else
                obj.topology = topology;
                obj.context = context;
            end

            obj.geometrySquare(); % Set default geometry to cartesian

            instance = obj;
        end

        function obj = setup(obj, global_size)
            % setup(global_resolution) establishes a global resolution &
            % runs the bookkeeping for it.
            if numel(global_size) == 2; global_size(3) = 1; end

            dblhalo = 2*obj.haloAmt;
            
            obj.pGlobalDomainRez = global_size;
            obj.pGlobalDomainRezPlusHalos   = global_size + dblhalo*double(obj.topology.nproc).*double(obj.topology.nproc > 1);

            % Default size: Ntotal / Nproc
            propSize = floor(obj.pGlobalDomainRezPlusHalos ./ double(obj.topology.nproc));
            
            % Compute offset
            obj.pLocalDomainOffset = (propSize-dblhalo).*double(obj.topology.coord);

            % If we're at the plus end of the topology in a dimension, increase proposed size to meet global domain resolution.
            for i = 1:obj.topology.ndim;
                if (double(obj.topology.coord(i)) == (obj.topology.nproc(i)-1)) && (propSize(i)*obj.topology.nproc(i) < obj.pGlobalDomainRezPlusHalos(i))
                    propSize(i) = propSize(i) + obj.pGlobalDomainRezPlusHalos(i) - propSize(i)*obj.topology.nproc(i);
                end
            end

            obj.pLocalRez = propSize;

            obj.edgeInterior(1,:) = double(obj.topology.coord > 0);
            obj.edgeInterior(2,:) = double(obj.topology.coord < (obj.topology.nproc-1));

            obj.circularBCs = [1 1 1];

            obj.updateGridVecs();

            instance = obj;
        end % Constructor

        function makeDimCircular(obj, dim)
            % makeDimCircular(1 <= dim <= 3) declares a circular BC on dim
            % Effect: Outer edge has a halo
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 1;
            obj.updateGridVecs();
        end

        function makeDimNotCircular(obj, dim)
            % makeDimNotCircular(1 <= dim <= 3) declares a noncircular BC on dim
            % Effect: Outer edge does not have a halo.
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 0;
            obj.updateGridVecs();
        end

        function geometrySquare(obj)
            obj.pGeometryType = ENUM.GEOMETRY_SQUARE;
            obj.pInnerRadius = 1.0;
        end

        function geometryCylindrical(obj, Rin)
            obj.pGeometryType = ENUM.GEOMETRY_CYLINDRICAL;
            %obj.makeDimNotCircular(1); % updates grid vecs
            obj.pInnerRadius  = Rin;
	    warning('WARNING: Cylindrical geometry does not reset rin per rank, do not use in parallel!!!');
	    % FIXME recompute pInnerRadius per rank based on localXvector(1).
        end
        
        function [u v w] = toLocalIndices(obj, x, y, z)
            % [u v w] = GIS.toLocalIndices(x, y, z) converts a global set of coordinates to 
            % local coordinates, and keeps only those in the local domain
            u = []; v = []; w = [];
            if (nargin == 2) & (size(x,2) == 3);
                z = x(:,3) - obj.pLocalDomainOffset(3);
                y = x(:,2) - obj.pLocalDomainOffset(2);
                x = x(:,1) - obj.pLocalDomainOffset(1);
  
                keep = (x>0) & (x<=obj.pLocalRez(1)) & (y>0) & (y<=obj.pLocalRez(2)) & (z>0) & (z<=obj.pLocalRez(3));
                u = [x(keep) y(keep) z(keep)];
                return;
            end
            if (nargin >= 2) & (~isempty(x)); x = x - obj.pLocalDomainOffset(1); u=x((x>0)&(x<=obj.pLocalRez(1))); end
            if (nargin >= 3) & (~isempty(y)); y = y - obj.pLocalDomainOffset(2); v=y((y>0)&(y<=obj.pLocalRez(2))); end
            if (nargin >= 4) & (~isempty(z)); z = z - obj.pLocalDomainOffset(3); w=z((z>0)&(z<=obj.pLocalRez(3))); end

        end

        function [x y z] = toCoordinates(obj, I0, Ix, Iy, Iz, h, x0)
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
        % Y = evaluateFunctionOnGrid(@func) calls afunc(x,y,z) using the
        % [x y z] returned by obj.ndgridSetXYZ.
            [x y z] = obj.ndgridSetXYZ();
            Y = afunc(x, y, z);
        end
        
        function localset = LocalIndexSet(obj, globalset, d)
        % Extracts the portion of ndgrid(1:globalsize(1), ...) visible to this node
        % Renders the 3 edge cells into halo automatically
        
            pLocalMax = obj.pLocalDomainOffset + obj.pLocalRez;

            if nargin == 3;
                localset =  q((q >= obj.pLocalDomainOffset(d)) & (q <= pLocalMax(d))) - obj.pLocalDomainOffset(d) + 1;
            else
                for n = 1:min(numel(obj.pGlobalDomainRezPlusHalos), numel(globalset));
                    q = globalset{n};
                    localset{n} = q((q >= obj.pLocalDomainOffset(n)) & (q <= pLocalMax(n))) - obj.pLocalDomainOffset(n) + 1;
                end
            end
        end
        
        function LL = cornerIndices(obj)
        % Return the index of the lower left corner in both local and global coordinates
        % Such that subtracting them from a 1:size(array) will make the lower-left corner that isn't part of the halo [0 0 0] in local coords and whatever it is in global coords
        ndim = numel(obj.pGlobalDomainRezPlusHalos);

            LL=[1 1 1; (obj.pLocalDomainOffset+1)]';
            for j = 1:ndim;
                LL(j,:) = LL(j,:) - 3*(obj.topology.nproc(j) > 1);
            end

        end

        function updateGridVecs(obj)
            % GIS.updateGridVecs(). Utility - upon change in global dims, recomputes x/y/z index
            % vectors
            ndim = numel(obj.pGlobalDomainRezPlusHalos);

            x=[]; lnh = [];
            for j = 1:ndim;
                q = 1:obj.pLocalRez(j);
                % This line degerates to the identity operation if nproc(j) = 1
                q = q + obj.pLocalDomainOffset(j) - 3*(obj.topology.nproc(j) > 1);

                % If the edges are periodic, wrap coordinates around
                if (obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1)
                    q = mod(q + obj.pGlobalDomainRez(j) - 1, obj.pGlobalDomainRez(j)) + 1;
                end
                x{j} = q;

                lmin = 1; lmax = obj.pLocalRez(j);
                if (obj.topology.coord(j) > 0) || ((obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1));
                    lmin = 4;
                end
                if (obj.topology.coord(j) < obj.topology.nproc(j)-1) || ((obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1));
                    lmax = lmax - 3;
                end

                lnh{j} = lmin:lmax;
            end

            if ndim == 2; x{3} = 1; end
        
            obj.localXvector = x{1}; obj.localYvector = x{2}; obj.localZvector = x{3};
            obj.nohaloXindex = lnh{1}; obj.nohaloYindex = lnh{2}; obj.nohaloZindex = lnh{3};
        end

        function [u v w] = ndgridVecs(obj)
            % [u v w] = GIS.ndgridVecs() returns the x-, y- and z- index vectors
            u = obj.localXvector; v = obj.localYvector; w = obj.localZvector;
        end

        function [x y z] = ndgridSetXYZ(obj, offset, scale)
            % [x y z] = ndgridsetXYZ(offset, scale) returns the part of
            % ndgrid( 1:grid(1), ...) that lives on this node.
            % Returns affine transform [Xn - offset(n)]*scale(n) if given,
            % defaulting to offset = [0 0 0] and scale = [1 1 1].
             
            if nargin > 1; % Lets the user get affine-transformed coordinates conveniently
                if nargin < 3; scale  = [1 1 1]; end;
                if nargin < 2; offset = [0 0 0]; end;
                [u v w] = obj.toCoordinates(offset, obj.localXvector, obj.localYvector, obj.localZvector, scale, [0 0 0]);
                [x y z] = ndgrid(u, v, w);
            else
                [x y z] = ndgrid(obj.localXvector, obj.localYvector, obj.localZvector);
                end
            
        end
        
        function [x y] = ndgridSetXY(obj, offset, scale)
            % See ndgridSetXYZ doc; Returns [x y] arrays. 
            % Note that offset/scale, if vector, must be 3 elements or
            % x(1)*[1 1 1] will be used instead
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1 1]; end;
                if nargin < 2; offset = [0 0 0]; end;
                [u v] = obj.toCoordinates(offset, obj.localXvector, obj.localYvector, [], scale, [0 0 0]);
                [x y] = ndgrid(u, v);
            else
                [x y] = ndgrid(obj.localXvector, obj.localYvector);
            end
        end
        
        function [y z] = ndgridSetYZ(obj, offset, scale)
            % See ndgridSetXYZ doc; Returns [y z] arrays. 
            % Note that offset/scale, if vector, must be 3 elements or
            % x(1)*[1 1 1] will be used instead
            [u v] = ndgrid(obj.localYvector, obj.localZvector);
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1]; end;
                if nargin < 2; offset = [0 0]; end;
                [u v w] = obj.toCoordinates(offset, obj.localXvector, obj.localYvector, obj.localZvector, scale, [0 0 0]);
                [y z] = ndgrid(v, w);
            else
                [y z] = ndgrid(obj.localYvector, obj.localZvector);
            end
        end
        
        function [x z] = ndgridSetXZ(obj, offset, scale)
            % See ndgridSetXYZ doc; Returns [x z] arrays. 
            % Note that offset/scale, if vector, must be 3 elements or
            % x(1)*[1 1 1] will be used instead
            [u v] = ndgrid(obj.localXvector, obj.localZvector);
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1]; end;
                if nargin < 2; offset = [0 0]; end;
                [u v w] = obj.toCoordinates(offset, obj.localXvector, obj.localYvector, obj.localZvector, scale, [0 0 0]);
                [x z] = ndgrid(u, w);
            else
                [x z] = ndgrid(obj.localXvector, obj.localZvector);
            end
        end

        % Generic function that the functions below talk to
        function out = makeValueArray(obj, dims, dtype, val)
            makesize = [];
            switch dims;
                case 1; makesize = [obj.pLocalRez(1) 1 1];
                case 2; makesize = [1 obj.pLocalRez(2) 1];
                case 3; makesize = [1 1 obj.pLocalRez(3)];
                case 4; makesize = [obj.pLocalRez(1:2) 1];
                case 5; makesize = [obj.pLocalRez(1) 1 obj.pLocalRez(3)];
                case 6; makesize = [1 obj.pLocalRez(2:3)];
                case 7; makesize = obj.pLocalRez;
            end
            
            % Slap a 3 on the first dim to build a vector
            % This is stupid and REALLY should be exchanged (3 goes LAST)
            if (nargin > 2) && (dtype == obj.VECTOR); makesize = [3 makesize]; end

            % Generate an array of 0 if not given a value
            if (nargin < 4); val = 0; end

            out = val * ones(makesize);
        end 

        % These generate a set of zeros of the size of the part of the global grid residing on this node
        function O = zerosXY(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(4, dtype, 0); end
        function O = zerosXZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(5, dtype, 0); end
        function O = zerosYZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(6, dtype, 0); end
        function O = zerosXYZ(obj, dtype); if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(7, dtype, 0); end

        % Or of ones
        function O = onesXY(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(4, dtype, 1); end
        function O = onesXZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(5, dtype, 1); end
        function O = onesYZ(obj, dtype);  if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(6, dtype, 1); end
        function O = onesXYZ(obj, dtype); if nargin < 2; dtype = obj.SCALAR; end; O = obj.makeValueArray(7, dtype, 1); end
        
        function [rho mom mag ener] = basicFluidXYZ(obj)
            rho = ones(obj.pLocalRez);
            mom = zeros([3 obj.pLocalRez]);
            mag = zeros([3 obj.pLocalRez]);
            ener= ones(obj.pLocalRez);
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

        % This is the only function in GIS that references any part of the topology execpt .nproc or .coord
        function index = getMyNodeIndex(obj)
            index = double(mod(obj.topology.neighbor_left+1, obj.topology.nproc));
        end

        function slim = withoutHalo(obj, array)
            % slim = GIS.withoutHalo(fat), when passed an array of size equal to the node's
            % array size, returns the array with halos removed.

            cantdo = 0;
            for n = 1:3; cantdo = cantdo || (size(array,n) ~= obj.pLocalRez(n)); end
                
            cantdo = mpi_max(cantdo); % All nodes must receive an array of the appropriate size
  
            if cantdo;
                disp(obj.topology);
                disp([size(array); obj.pLocalRez]);
                error('Oh teh noez, rank %i received an array of invalid size!', mpi_myrank());
            end
      
            slim = array(obj.nohaloXindex, obj.nohaloYindex, obj.nohaloZindex);
        end

        function DEBUG_setTopoSize(obj, n);
            obj.topology.nproc = n;

            c = obj.circularBCs;
            obj.setup(obj.pGlobalDomainRez);
            for x = 1:n; if c(x) == 0; obj.makeDimNotCircular(x); end; end
        end
        function DEBUG_setTopoCoord(obj,c)
            obj.topology.coord = c;
            c = obj.circularBCs;
            obj.setup(obj.pGlobalDomainRez);
            for x = 1:n; if c(x) == 0; obj.makeDimNotCircular(x); end; end
        end

    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

