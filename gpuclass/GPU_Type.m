classdef GPU_Type < handle
% This is the new master GPU class
% It provides a barebones handle for GPU objects in Matlab
% Also, very very importantly, it TRUSTS YOU WITH THE BLOODY POINTER
% If matlab's support were not a douche on that very point this would not exist

    properties (Constant = true, Transient = true)

    end

    properties (SetAccess = public, GetAccess = public, Transient = true)

    end % Public

    properties (SetAccess = private, GetAccess = private, Transient= true)
        allocated; % true if this GPU pointer was actually alloc'd and must be freed-on-quit
                   % false if it's simply pointed at another GPU_Type
        maxNumel;  % Indicates linear size of array that can be written
    end % Private

    properties (Dependent = true)
        array; % Accessing copies GPU array to CPU matlab array & returns it
    end % Dependent

    properties (SetAccess = private, GetAccess = public)
        GPU_MemPtr; % A 5x1 array of int64_ts:
                    % x[1:5] = [gpu memory pointer, # of dimensions, Nx,
                    % Ny, Nz]. Defaults to [0 0 0 0 0 0] if not initialized.
        asize;      % The size of the array. Posesses 3 elements.
        numdims;    % integer, 2 or 3 depending on how many nonzero extents exist
                    % This is for compatibility with matlab,
                    % even if it is mathematically nonsense to claim [] has 2 dimensions
        manager;
    end

    methods
        function obj = GPU_Type(arrin, docloned)
            obj.allocated = false;
            obj.clearArray();
            if nargin == 1; docloned = 0; end

            if nargin > 0
                obj.handleDatain(arrin, docloned);
            end

            obj.manager = GPUManager.getInstance();

        end % Constructor

        function delete(obj)
            if(obj.allocated == true) GPU_free(obj); end
        end % Destructor

        function set.array(obj, arrin)
            % Goofus doesn't care if he leaks memory
            % Gallant always cleans up after himself
            if obj.allocated == true; GPU_free(obj); obj.allocated = false; end
            obj.handleDatain(arrin, 0);

        end

        function result = get.array(obj)
            % Return blank if not allocated, or dump the GPU array to CPU if we are
            if obj.allocated == false; result = []; return; end
            
            result = GPU_download(obj.GPU_MemPtr);
        end

        function result = eq(obj)
            result = GPU_Type(obj);
        end

        % Updates matlab-facing properties in event of a replace-in-place function
        % This hopefully avoids the slowness of the set.array function
        function flushTag(obj)
            q = double(obj.GPU_MemPtr);
            obj.numdims = 3; if obj.asize(3) == 1; obj.numdims = 2; end
            obj.asize = q(1:3)';
        end

        function result = size(obj, dimno)
            if nargin == 1
                if obj.numdims == 2; result = obj.asize(1:2); else result = obj.asize; end
            else
                result = obj.asize(dimno);
            end
        end

        function result = ndims(obj)
            result = obj.numdims;
        end

        function result = numel(obj)
            result = prod(obj.asize);
        end

        % Cookie-cutter operations for basic math interpertation
        % Warning, these are very much suboptimal due to excessive memory BW use        
        function y = plus(a, b)
            y = GPU_Type(cudaBasicOperations(a, b, 1));
        end

        function y = minus(a,b)
            y = GPU_Type(cudaBasicOperations(a, b, 2));
        end

        function y = mtimes(a,b); y = times(a,b); end
        function y = times(a,b)
            y = GPU_Type(cudaBasicOperations(a, b, 3));
        end
        function y = rdivide(a,b)
            y = GPU_Type(cudaBasicOperations(a, b, 4));
        end

        function y = min(a,b)
            y = GPU_Type(cudaBasicOperations(a, b, 5));
        end

        function y = max(a,b)
            y = GPU_Type(cudaBasicOperations(a, b, 6));
        end

        % This is pretty much a hack, in place until we do the magnetic operations properly.
        function y = harmonicmean(a, b); y = GPU_Type(cudaBasicOperations(a, b, 7)); end

        function y = sqrt(a); y = GPU_Type(cudaBasicOperations(a,1)); end
        function y = log(a);  y = GPU_Type(cudaBasicOperations(a,2)); end
        function y = exp(a);  y = GPU_Type(cudaBasicOperations(a,3)); end
        function y = sin(a);  y = GPU_Type(cudaBasicOperations(a,4)); end
        function y = cos(a);  y = GPU_Type(cudaBasicOperations(a,5)); end
        function y = tan(a);  y = GPU_Type(cudaBasicOperations(a,6)); end
        function y = asin(a);  y = GPU_Type(cudaBasicOperations(a,7)); end
        function y = acos(a);  y = GPU_Type(cudaBasicOperations(a,8)); end
        function y = atan(a);  y = GPU_Type(cudaBasicOperations(a,9)); end
        function y = sinh(a);  y = GPU_Type(cudaBasicOperations(a,10)); end
        function y = cosh(a);  y = GPU_Type(cudaBasicOperations(a,11)); end
        function y = tanh(a);  y = GPU_Type(cudaBasicOperations(a,12)); end
        function y = asinh(a);  y = GPU_Type(cudaBasicOperations(a,13)); end
        function y = acosh(a);  y = GPU_Type(cudaBasicOperations(a,14)); end
        function y = atanh(a);  y = GPU_Type(cudaBasicOperations(a,15)); end

        function y = transpose(a); y = GPU_Type(cudaArrayRotateB(a,2)); end
        function y = Ztranspose(a); y = GPU_Type(cudaArrayRotateB(a,3)); end
        function y = YZtranspose(a); y = GPU_Type(cudaArrayRotateB(a,4)); end

        function clearArray(obj)
            if obj.allocated; GPU_free(obj.GPU_MemPtr); end
            obj.allocated = false;
            obj.GPU_MemPtr = int64(zeros([1 10]));
            obj.asize = [0 0 0];
            obj.numdims = 2;
        end

        function makeBCHalos(self, halos)
            bits = 1*halos(1,1) + 2*halos(2,1) + 4*halos(1,2) + 8*halos(2,2) + 16*halos(1,3) + 32*halos(2,3);
            self.GPU_MemPtr(10) = int64(bits);
        end

        % Convert this GPU_Type into a slab; This requires a reallocation & thus a new tag.
        function createSlabs(obj, N)
            B = GPU_makeslab(obj.GPU_MemPtr, N);
            obj.GPU_MemPtr = B;
        end

    end % generic methods

    methods (Access = private)

        function handleDatain(obj, arrin, docloned)
            gm = GPUManager.getInstance();
            
            if obj.GPU_MemPtr(4) < 0
                % This is a slab handle: we cannot safely free or resize it here
                % We must copy the input over the original data 
                if isa(arrin, 'double')
                    % We need to transfer this to the GPU first
                    halo = gm.useHalo;
                    if docloned;
                        halo = 0;
                        pd = gm.partitionDir;
                        
                        if (size(arrin,pd) ~= 1) && (size(arrin,pd) ~= numel(gm.deviceList));
                            error('Upload of cloned data does not fit.');
                        end
                    end
                    
                    tmpPtr = GPU_upload(arrin, gm.deviceList, [halo gm.partitionDir (gm.nprocs(gm.partitionDir) == 1) ]);
                else
                    tmpPtr = arrin;
                end
                
                % Do the transfer
                GPU_copy(obj.GPU_MemPtr, tmpPtr);
                
                % If it came from a CPU array, dump the copy
                if isa(arrin, 'double'); GPU_free(tmpPtr); end
                obj.allocated = 1; 
            else
                if isa(arrin, 'double')
                    if isempty(arrin); obj.clearArray(); return; end
                    
                    % Cast a CPU double to a GPU double
                    obj.allocated = true;
                    obj.asize = size(arrin);
                    if numel(obj.asize) == 2; obj.asize(3) = 1; end
                    obj.numdims = ndims(arrin);
                    
                    halo = gm.useHalo;
                    if docloned;
                        halo = 0;
                        pd = gm.partitionDir;
                        
                        if (size(arrin,pd) ~= 1) && (size(arrin,pd) ~= numel(gm.deviceList));
                            error('Upload of cloned data does not fit.');
                        end
                    end
                    
                    obj.GPU_MemPtr = GPU_upload(arrin, gm.deviceList, [halo gm.partitionDir gm.useExterior]);
                elseif isa(arrin, 'GPU_Type') == 1
                    obj.allocated = true;
                    obj.asize     = arrin.asize;
                    obj.numdims   = arrin.numdims;
                    
                    obj.GPU_MemPtr = GPU_clone(arrin);
                elseif (isa(arrin, 'int64') == 1)
                    % Convert a gpu routine-returned 5-int tag to a GPU_Type for matlab
                    obj.allocated = 1;
                    obj.GPU_MemPtr = arrin;
                    
                    q = double(arrin);
                    obj.numdims = 3; if q(3)==1; obj.numdims = 2; end
                    obj.asize = q(1:3)';
                else
                    error('GPU_Type must be set with either a double array, another GPU_Type, or int64 tag returned by gpu routine');
                end
            end
            
        end

    end % Private methods

    methods (Static = true)

    end % Static methods

end

