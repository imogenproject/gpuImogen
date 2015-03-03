classdef GlobalIndexSemantics < handle
% Global Index Semantics: Translates global index requests to local ones for lightweight global array support
% x = GlobalIndexSemantics(context, topology): initialize
% TF = GlobalIndexSemantics('dummy'): Check if previous init'd
% x = GlobalIndexSemantics(): Retreive

    properties (Constant = true, Transient = true)
    end

    properties (SetAccess = public, GetAccess = public, Transient = true)
    end % Public

    properties (SetAccess = private, GetAccess = public)
        context;
        topology;
        pGlobalDims; % The size of global input domain size + halo added on [e.g. 512 500]
        pMySize;     % The size of the local domain + any added halo [256 500
        pMyOffset;   % The offset of the local domain taking the lower left corner of the
                     % input domain (excluding halo).
        pHaloDims;   % The input global domain size, with no halo added [e.g. 500 500]

        edgeInterior; % Whether my [ left or right, x/y/z ] side is interior & therefore circular or exterior

        domainResolution;
        domainOffset;

    end % Private

    properties (SetAccess = private, GetAccess = private)
        localXvector; localYvector; localZvector; circularBCs;
    end

    properties (Dependent = true)
    end % Dependent

    methods
        function obj = GlobalIndexSemantics(context, topology)
            persistent instance;

            if ~(isempty(instance) || ~isvalid(instance))
                obj = instance; return;
            end

            if nargin < 2;
                warning('desire GlobalIndexSemantics(parallelContext, parallelTopology): Fudging scalar operation...');
                [obj.context obj.topology] = fakeParallelStart();
            else
                obj.topology = topology;
                obj.context = context;
            end

            instance = obj;
        end

        function obj = setup(obj, global_size)

            obj.pGlobalDims = global_size + 6*double(obj.topology.nproc).*double(obj.topology.nproc > 1);

            pMySize = floor(obj.pGlobalDims ./ double(obj.topology.nproc));
            
            % Make up any deficit in size with the rightmost nodes in each dimension
            for i = 1:obj.topology.ndim;
                if (double(obj.topology.coord(i)) == (obj.topology.nproc(i)+1)) && (pMySize(i)*obj.topology.nproc(i) < obj.pGlobalDims(i))
                    pMySize(i) = pMySize(i) + obj.pGlobalDims(i) - pMySize(i)*obj.topology.nproc(i);
                end
            end
            obj.pMySize = pMySize;
            obj.pMyOffset = (obj.pMySize-6).*double(obj.topology.coord);

            obj.pHaloDims = obj.pGlobalDims - (obj.topology.nproc > 1).*double((6*obj.topology.nproc));

            obj.edgeInterior(1,:) = double(obj.topology.coord > 0);
            obj.edgeInterior(2,:) = double(obj.topology.coord < (obj.topology.nproc-1));

            obj.domainResolution = global_size;
            obj.domainOffset     = obj.pMyOffset;% - double(6*obj.topology.coord);

            obj.circularBCs = [1 1 1];

            obj.updateGridVecs();

            instance = obj;
        end % Constructor

        % Assigns the given dimension to be circular, enabling edge sharing across the outer boundary
        function makeDimCircular(obj, dim)
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 1;
            obj.updateGridVecs();
        end

        function makeDimNotCircular(obj, dim)
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.circularBCs(dim) = 0;
            obj.updateGridVecs();
        end



        % Converts a global set of coordinates to local coordinates, and keeps only those in the local domain
        function [u v w] = toLocalIndices(obj, x, y, z)
            u = []; v = []; w = [];
            if (nargin >= 2) & (~isempty(x)); x = x - obj.pMyOffset(1); u=x((x>0)&(x<=obj.pMySize(1))); end
            if (nargin >= 3) & (~isempty(y)); y = y - obj.pMyOffset(2); v=y((y>0)&(y<=obj.pMySize(2))); end
            if (nargin >= 4) & (~isempty(z)); z = z - obj.pMyOffset(3); w=z((z>0)&(z<=obj.pMySize(3))); end
        end

	function [x y z] = toCoordinates(obj, I0, Ix, Iy, Iz, h, x0)
	% [x y z] = toCoordinates(obj, I0, Ix, Iy, Iz, h, x0) returns (for n = {x, y, z})
	% (In - I0(n))*h(n) - x0(n), i.e.
	% I0 is an index offset, h the coordinate spacing and x0 the coordinate offset.
	% [] for I0 and x0 default to zero; [] for h defaults to 1; scalars are multiplied by [1 1 1]

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

        % Extracts the portion of ndgrid(1:globalsize(1), ...) visible to this node
        % Renders the 3 edge cells into halo automatically
        function localset = LocalIndexSet(obj, globalset, d)
            localset = [];
            pLocalMax = obj.pMyOffset + obj.pMySize;

            if nargin == 3;
                localset =  q((q >= obj.pMyOffset(d)) & (q <= pLocalMax(d))) - obj.pMyOffset(d) + 1;
            else
                for n = 1:min(numel(obj.pGlobalDims), numel(globalset));
                    q = globalset{n};
                    localset{n} = q((q >= obj.pMyOffset(n)) & (q <= pLocalMax(n))) - obj.pMyOffset(n) + 1;
                end
            end
        end

        % Return the index of the lower left corner in both local and global coordinates
        % Such that subtracting them from a 1:size(array) will make the lower-left corner that isn't part of the halo [0 0 0] in local coords and whatever it is in global coords
        function LL = cornerIndices(obj)
            ndim = numel(obj.pGlobalDims);

            LL=[1 1 1; (obj.pMyOffset+1)]';
            for j = 1:ndim;
                LL(j,:) = LL(j,:) - 3*(obj.topology.nproc(j) > 1);
            end

        end

        % Gets the 1xN vectors containing 1:N_i for all 3 dimensions
        function updateGridVecs(obj)
            ndim = numel(obj.pGlobalDims);

            x=[];
            for j = 1:ndim;
                q = 1:obj.pMySize(j);
                % This line degerates to the identity operation if nproc(j) = 1
                q = q + obj.pMyOffset(j) - 3*(obj.topology.nproc(j) > 1);

                % If the edges are periodic, wrap coordinates around
                if (obj.topology.nproc(j) > 1) && (obj.circularBCs(j) == 1)
                    q = mod(q + obj.pHaloDims(j) - 1, obj.pHaloDims(j)) + 1;
                end
                x{j} = q;
            end

            if ndim == 2; x{3} = 1; end
        
            obj.localXvector = x{1}; obj.localYvector = x{2}; obj.localZvector = x{3};       
        end

        function [u v w] = ndgridVecs(obj)
            u = obj.localXvector; v = obj.localYvector; w = obj.localZvector;
        end

        % These return the part of ndgrid(1:globalsize(1), ...) that would reside on this node
        % Halo is automatically part of it.
        function [x y z] = ndgridSetXYZ(obj, offset, scale)
            [u v w] = ndgrid(obj.localXvector, obj.localYvector, obj.localZvector);
            
            if nargin > 1; % Lets the user get affine-transformed coordinates conveniently
                if nargin < 3; scale  = [1 1 1]; end;
                if nargin < 2; offset = [0 0 0]; end;
                [x y z] = obj.toCoordinates(offset, u, v, w, scale, [0 0 0]);
            else
                x = u; y = v; z = w;
            end
            
        end
        
        function [x y] = ndgridSetXY(obj, offset, scale)
            [u v] = ndgrid(obj.localXvector, obj.localYvector);
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1]; end;
                if nargin < 2; offset = [0 0]; end;
                [x y] = obj.toCoordinates(offset, u, v, [], scale, [0 0 0]);
            else
                x = u; y = v;
            end
        end
        
        function [y z] = ndgridSetYZ(obj)
            [u v] = ndgrid(obj.localYvector, obj.localZvector);
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1]; end;
                if nargin < 2; offset = [0 0]; end;
                [y z] = obj.toCoordinates(offset, u, v, [], scale, [0 0 0]);
            else
                y = u; z = v;
            end
        end
        
        function [x z] = ndgridSetXZ(obj)
            [u v] = ndgrid(obj.localXvector, obj.localZvector);
            
            if nargin > 1;
                if nargin < 3; scale  = [1 1]; end;
                if nargin < 2; offset = [0 0]; end;
                [x z] = obj.toCoordinates(offset, u, v, [], scale, [0 0 0]);
            else
                x = u; z = v;
            end
        end

        % These generate a set of ones the size of the part of the global grid residing on this node
        function O = onesSetXY(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(b)]); end
        function O = onesSetYZ(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(b) numel(c)]); end
        function O = onesSetXZ(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(c)]); end
        function O = onesSetXYZ(obj); [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(b) numel(c)]); end

        % Function returns an nx x ny x nz array whose (i,j,k) index contains the rank of the node residing on the
        % nx by ny by nz topology at that location.
        function G = getNodeGeometry(obj)
            G = zeros(obj.topology.nproc);
            xi = mpi_allgather( obj.topology.coord(1) );
            yi = mpi_allgather( obj.topology.coord(2) );
            zi = mpi_allgather( obj.topology.coord(3) );
            i0 = xi + obj.topology.nproc(1)*(yi + obj.topology.nproc(2)*zi) + 1;
            G(i0) = 0:(numel(G)-1);
        end

        function index = getMyNodeIndex(obj)
            index = double(mod(obj.topology.neighbor_left+1, obj.topology.nproc));
        end

        % Given a set of values to index a global-sized array, restrict it to those within the local domain.
        %function [out index] = findIndicesInMyDomain(obj, in, indexMode)
        %    if (nargin == 2) || (indexMode == 1); in = in - 1; end % convert to zero-based indexing

            % Translate indices back to x-y-z coordinates.
        %    delta = obj.pGlobalDims(1)*obj.pGlobalDims(2);
        %    z = (in - mod(in, delta))/delta;
        %    in = in - z*delta;
        %    delta = obj.pGlobalDims(1);
        %    y = (in - mod(in, delta))/delta;
        %    x = in - y*delta;

        %    i0 = find(

        %end        

    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

