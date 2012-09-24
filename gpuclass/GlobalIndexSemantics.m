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

        nodeIndex;
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

            obj.pGlobalDims = global_size;

            pMySize = floor(obj.pGlobalDims ./ double(obj.topology.nproc));
            
            obj.nodeIndex = double(mod(obj.topology.neighbor_left + 1, obj.topology.nproc));

            % Make up any deficit in size with the rightmost nodes in each dimension
            for i = 1:obj.topology.ndim;
                if (obj.nodeIndex(i) == (obj.topology.nproc(i)+1)) && (pMySize(i)*obj.topology.nproc(i) < obj.pGlobalDims(i))
                    pMySize(i) = pMySize(i) + obj.pGlobalDims(i) - pMySize(i)*obj.topology.nproc(i);
                end
            end
            obj.pMySize = pMySize;
            obj.pMyOffset = obj.pMySize.*obj.nodeIndex;

            obj.pHaloDims = obj.pGlobalDims - (obj.topology.nproc > 1).*double((6*obj.topology.nproc));

            instance = obj;
        end % Constructor

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
                q = q + (obj.pMySize(j) - 6*(obj.topology.nproc(j) > 1))*obj.nodeIndex(j) - 3*(obj.topology.nproc(j) > 1) - 1;
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


    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

