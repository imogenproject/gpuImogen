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
        allocated; % true if GPU pointer is valid, false at start.
    end % Private

    properties (Dependent = true)
        array; % Accessing copies GPU array to CPU matlab array & returns it
    end % Dependent

    properties (SetAccess = private, GetAccess = public)
        GPU_MemPtr; % A 5x1 array of int64_ts:
                    % x[1:5] = [gpu memory pointer, # of dimensions, Nx,
                    % Ny, Nz]. Defaults to [0 0 0 0 0] if not initialized.
        asize;      % The size of the array. Posesses 3 elements.
        numdims;    % integer, 2 or 3 depending on how many nonzero extents exist
                    % This is for compatibility with matlab,
                    % even if it is mathematically nonsense to claim [] has 2 dimensions
    end

    methods
        function obj = GPU_Type(arrin)
            obj.allocated = false;
            obj.clearArray();

            if nargin > 0
                obj.handleDatain(arrin);
            end

        end % Constructor

        function delete(obj)
            if(obj.allocated == true) GPU_free(obj.GPU_MemPtr); end
        end % Destructor

        function set.array(obj, arrin)
            % Goofus doesn't care if he leaks memory
            % Gallant always cleans up after himself
            if obj.allocated == true; GPU_free(obj.GPU_MemPtr); obj.allocated = false; end
            obj.handleDatain(arrin);

        end

        function result = get.array(obj)
            % Return blank if not allocated, or dump the GPU array to CPU if we are
            if obj.allocated == false; result = []; return; end
            result = GPU_cudamemcpy(obj.GPU_MemPtr);
        end

        function result = eq(obj)
            result = GPU_Type(obj);
        end

        function result = size(obj, dimno)
            if nargin == 1
                if obj.numdims == 2; result = obj.asize(1:2); else; result = obj.asize; end
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
        function y = plus(a, b);
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end

            y = GPU_Type(cudaBasicOperations(q, r, 1)); return;

%            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 1)); return;  end
%            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 1)); return; end;
%            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 1)); return; end;
        end

        function y = minus(a,b)
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end
            y = GPU_Type(cudaBasicOperations(q, r, 2)); return;

            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 2)); return; end
            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 2)); return; end;
            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 2)); return; end;
        end

        function y = mtimes(a,b); y = times(a,b); end
        function y = times(a,b)
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end
            y = GPU_Type(cudaBasicOperations(q, r, 3)); return;

            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 3)); return; end
            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 3)); return; end;
            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 3)); return; end;
        end
        function y = rdivide(a,b)
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end
            y = GPU_Type(cudaBasicOperations(q, r, 4)); return;

            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 4)); return; end
            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 4)); return; end;
            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 4)); return; end;
        end

        function y = min(a,b)
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end
            y = GPU_Type(cudaBasicOperations(q, r, 5)); return;

            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 5)); return; end
            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 5)); return; end;
            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 5)); return; end;
        end

        function y = max(a,b)
            if isa(a, 'GPU_Type'); q = a.GPU_MemPtr; else; q = a; end
            if isa(b, 'GPU_Type'); r = b.GPU_MemPtr; else; r = b; end
            y = GPU_Type(cudaBasicOperations(q, r, 6)); return;

            if isa(a, 'GPU_Type') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 6)); return; end
            if isa(a, 'GPU_Type') && isa(b, 'double'); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b, 6)); return; end;
            if isa(a, 'double') && isa(b, 'GPU_Type'); y = GPU_Type(cudaBasicOperations(a, b.GPU_MemPtr, 6)); return; end;
        end

        % This is pretty much a hack, in place until we do the magnetic operations properly.
        function y = harmonicmean(a, b); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr, b.GPU_MemPtr, 7)); return; end

        function y = sqrt(a); y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,1)); return; end
        function y = log(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,2)); return; end
        function y = exp(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,3)); return; end
        function y = sin(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,4)); return; end
        function y = cos(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,5)); return; end
        function y = tan(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,6)); return; end
        function y = asin(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,7)); return; end
        function y = acos(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,8)); return; end
        function y = atan(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,9)); return; end
        function y = sinh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,10)); return; end
        function y = cosh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,11)); return; end
        function y = tanh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,12)); return; end
        function y = asinh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,13)); return; end
        function y = acosh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,14)); return; end
        function y = atanh(a);  y = GPU_Type(cudaBasicOperations(a.GPU_MemPtr,15)); return; end

        function y = transpose(a); y = GPU_Type(cudaArrayRotate(a.GPU_MemPtr,2)); return; end
        function y = Ztranspose(a); y = GPU_Type(cudaArrayRotate(a.GPU_MemPtr,3)); return; end

        function clearArray(obj)
            if obj.allocated== true; GPU_free(obj.GPU_MemPtr); end
            obj.allocated = false;
            obj.GPU_MemPtr = int64([0 0 0 0 0 0]);
            obj.asize = [0 0 0];
            obj.numdims = 2;
        end

    end % generic methods

    methods (Access = private)

        function handleDatain(obj, arrin)
            if isa(arrin, 'double')
                if isempty(arrin); obj.clearArray(); return; end

                % Cast a CPU double to a GPU double
                obj.allocated = true;
                obj.asize = size(arrin);
                if numel(obj.asize) == 2; obj.asize(3) = 1; end
                obj.numdims = ndims(arrin);

                obj.GPU_MemPtr = GPU_cudamemcpy(arrin);
            elseif isa(arrin, 'GPU_Type') == 1
                % Make a copy of another GPU double
                obj.allocated = true;
                obj.asize     = arrin.asize;
                obj.numdims   = arrin.numdims;

                obj.GPU_MemPtr = GPU_cudamemcpy(arrin.GPU_MemPtr, 1);
            elseif (isa(arrin, 'int64') == 1) && (numel(arrin) == 5)
                % Convert a gpu routine-returned 5-int tag to a GPU_Type for matlab
                obj.allocated = true;
                obj.GPU_MemPtr = arrin;

                q = double(arrin);
                obj.numdims = q(2);
                obj.asize = q(3:5)';
            else
                 error('GPU_Type must be set with either a double array, another GPU_Type, or 5-int64 tag from gpu routine');
            end
        end

    end % Private methods

    methods (Static = true)

    end % Static methods

end

