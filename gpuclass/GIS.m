classdef GIS < handle
% Global Index Semantics: Translates global index requests to local ones for lightweight global array support

    properties (Constant = true, Transient = true)
    end

    properties (SetAccess = public, GetAccess = public, Transient = true)

    end % Public

    properties (SetAccess = private, GetAccess = public)
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
        function obj = GIS(global_size, context, topology)
            if nargin ~= 3; error('GIS(global_size, parallelContext, parallelTopology): all args required'); end

            obj.topology = topology;
            obj.pGlobalDims = global_size;

            pMySize = floor(obj.pGlobalDims ./ double(topology.nproc));
            
            obj.nodeIndex = double(mod(topology.neighbor_left + 1, topology.nproc));

            % Make up any deficit in size with the rightmost nodes in each dimension
            for i = 1:topology.ndim;
                if (obj.nodeIndex(i) == (topology.nproc(i)+1)) && (pMySize(i)*topology.nproc(i) < obj.pGlobalDims(i))
                    pMySize(i) = pMySize(i) + obj.pGlobalDims(i) - pMySize(i)*topology.nproc(i);
                end
            end
            obj.pMySize = pMySize;
            obj.pMyOffset = obj.pMySize.*obj.nodeIndex;

            obj.pHaloDims = obj.pGlobalDims - (topology.nproc > 1).*double((6*topology.nproc));
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

       function [u v w] = ndgridSet(obj)
           if numel(obj.pGlobalDims) == 3
               [u v w] = ndgrid(1:obj.pMySize(1), 1:obj.pMySize(2), 1:obj.pMySize(3)); % Create ndgrid the size of the entire tile
           else
               [u v] = ndgrid(1:obj.pMySize(1), 1:obj.pMySize(2));
           end

           % Advance by the size of the non-halo'd tile's index
           u = u + (obj.pMySize(1)-6*(obj.topology.nproc(1) > 1))*obj.nodeIndex(1) - 3*(obj.topology.nproc(1) > 1) - 1;
           v = v + (obj.pMySize(2)-6*(obj.topology.nproc(2) > 1))*obj.nodeIndex(2) - 3*(obj.topology.nproc(2) > 1) - 1;

           % Wrap negative and overflow indices around
           u(u < 0) = u(u < 0) + obj.pHaloDims(1);
           v(v < 0) = v(v < 0) + obj.pHaloDims(2);

           u = mod(u, obj.pHaloDims(1)) + 1;
           v = mod(v, obj.pHaloDims(2)) + 1;

           if numel(obj.pGlobalDims) == 3
               w = w + (obj.pMySize(3)-6*(obj.topology.nproc(3) > 1))*obj.nodeIndex(3) - 3*(obj.topology.nproc(3) > 1) - 1;
               w(w < 0) = w(w < 0) + obj.pHaloDims(3);
               w = mod(w, obj.pHaloDims(3)) + 1;
           else
               w = [];
           end

       end


    end % generic methods

    methods (Access = private)

    end % Private methods

    methods (Static = true)

    end % Static methods

end

