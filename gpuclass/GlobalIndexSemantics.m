classdef GlobalIndexSemantics < handle
% Global Index Semantics: Translates global index requests to local ones for lightweight global array support

    properties (Constant = true, Transient = true)
    end

    properties (SetAccess = public, GetAccess = public, Transient = true)

    end % Public

    properties (SetAccess = private, GetAccess = public)
        context;
        topology;
        pGlobalDims; % The size of the entire global array in total (halo-included) coordinates
        
        pMySize; pMyOffset; % Size/osset for total (halo-included) array
        % i.e. if pGlobalDims = [256 256] and we have a [2 4] processor topology,
        % pMySize will report [128 64] and pMyOffset will be multiples of [128 64],
        % even though only [122 58] cells are usable per tile with 6 lost to halo.

        pHaloDims;

        edgeInterior; % Whether my [ left or right, x/y/z ] side is interior & therefore circular or exterior

        domainResolution;
    end % Private

    properties (Dependent = true)
    end % Dependent

    methods
        function obj = GlobalIndexSemantics(context, topology)
            persistent instance;
            if ~(isempty(instance) || ~isvalid(instance))
                obj = instance; return;
            end

            if nargin ~= 2; error('GlocalIndexSemantics(parallelContext, parallelTopology): all args required'); end

            obj.topology = topology;
            obj.context = context;

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
            obj.pMyOffset = obj.pMySize.*double(obj.topology.coord);

            obj.pHaloDims = obj.pGlobalDims - (obj.topology.nproc > 1).*double((6*obj.topology.nproc));

            obj.edgeInterior(1,:) = double(obj.topology.coord > 0);
            obj.edgeInterior(2,:) = double(obj.topology.coord < (obj.topology.nproc-1));

            obj.domainResolution = global_size;

            instance = obj;
        end % Constructor

        function makeDimCircular(obj, dim)
            if (dim < 1) || (dim > 3); error('Dimension must be between 1 and 3\n'); end
            obj.edgeInterior(:,dim) = 1;
        end

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

        function [u v w] = ndgridVecs(obj)
            ndim = numel(obj.pGlobalDims);

            x=[];
            for j = 1:ndim;
                q = 1:obj.pMySize(j);
                q = q + (obj.pMySize(j) - 6*(obj.topology.nproc(j) > 1))*double(obj.topology.coord(j)) - 3*(obj.topology.nproc(j) > 1) - 1;
                q(q < 0) = q(q < 0) + obj.pHaloDims(j);
                x{j} = mod(q, obj.pHaloDims(j)) + 1;
            end

            if ndim == 2; x{3} = 1; end
        
            u = x{1}; v = x{2}; w = x{3};       
        end

        function [u v w] = ndgridSetXYZ(obj)
            [a b c] = obj.ndgridVecs();
            [u v w] = ndgrid(a, b, c);
        end

        function [x y] = ndgridSetXY(obj)
            [a b c] = obj.ndgridVecs();
            [x y] = ndgrid(a, b);
        end

        function [y z] = ndgridSetYZ(obj)
            [a b c] = obj.ndgridVecs();
            [y z] = ndgrid(b, c);
        end

        function [x z] = ndgridSetXZ(obj)
            [a b c] = obj.ndgridVecs();
            [x z] = ndgrid(a, c);
        end

        function O = onesSetXY(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(b)]); end
        function O = onesSetYZ(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(b) numel(c)]); end
        function O = onesSetXZ(obj);  [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(c)]); end
        function O = onesSetXYZ(obj); [a b c] = obj.ndgridVecs(); O = ones([numel(a) numel(b) numel(c)]); end

    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

