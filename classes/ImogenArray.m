classdef ImogenArray < handle
    % The base class for all Imogen array objects. Provides all of the common functionality shared by
    % the flux-source array objects.
    
    %===================================================================================================
    properties (GetAccess = public, Constant = true, Transient = true) %        C O N S T A N T  [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        component;      % Vector component of the array (0,1,2,3).                  int
        id;             % Identifier specifying the object data type.               cell(1,?)
        bcModes;        % Boundary condition types for each direction.              cell(2,3)
        bcHaloShare;    % Whether to take part in the halo exchange                 double 2x3
        edgeStore;      % Stored edge values for shifting.                          Edges
        
        boundaryData;   % Store boundary condition data during runtime              struct
        
        staticValues;   % Values for static array indices.                          double
        staticCoeffs;   % Coefficients used to determine how fast the array fades to the value double
        staticIndices;  % Indices of staticValues used by StaticArray.                int
        staticOffsets;  % Offsets to access the permutations of the statics arrays
        
        edgeshifts;     % Handles to shifting functions for each grid direction.    handle(2,3)
        isZero;         % Specifies that the array is statically zero.              logical
    end %PROPERTIES
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pArray;         % Storage of the 3D array data.                             double(Nx,Ny,Nz)
        pShiftEnabled;  % Specifies if shifting is active for each dimension.       logical(3)
        pManager;  % Access to the FluidManager associated with this state vector FluidManager
        
        pBCUninitialized; % Specifies if obj has been initialized.                    bool
    end %PROPERTIES
    
    %===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        gridSize;       % Size of the data array.                                   int(3)
        array;          % Storage of the 3D array data: copied out of gpu memory on demand double(Nx,Ny,Nz)
        gputag;         % GPU tag of the array for passing to cuda
        streamptr;      % .pArray.manager.cudaStreamsPtr
        idString;       % String form of the id cell array.                         str
        fades;          % The fade objects influencing the ImogenArray object.      cell{?}
    end %DEPENDENT
    
    
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        %___________________________________________________________________________________________________ GS: array
        % Main property accessor to the data array for the ImogenArray object.
        function result = get.array(obj)
            if isa(obj.pArray,'GPU_Type');
                result = obj.pArray.array;
            else
                result = obj.pArray;
            end
        end
        
        function result = get.gputag(obj)
            result = obj.pArray.GPU_MemPtr;
        end
        function result = get.streamptr(obj); result = obj.pArray.manager.cudaStreamsPtr; end
        
        function initialArray(obj, array)
            obj.pArray.array = array;
            if obj.pBCUninitialized;
                % Make certain everyone is on board & shares the same view before setting up BCs
                parallels = ParallelGlobals();
                
                % This reads the Matlab class's bcHaloShare and sets the bits in the MGArray structure
                obj.pArray.makeBCHalos(obj.bcHaloShare);
                
                % Make sure we have a globally coherent view before going any further
                cudaHaloExchange(obj, 1, parallels.topology, obj.bcHaloShare);
                cudaHaloExchange(obj, 2, parallels.topology, obj.bcHaloShare);
                cudaHaloExchange(obj, 3, parallels.topology, obj.bcHaloShare);
                
                obj.setupBoundaries();
                obj.pBCUninitialized = false;
            else % Initialize if needed.
                warning('Warning: obj.initialArray() called when array already initialized.');
            end
            
            for d = 1:3;
                if obj.gridSize(d) > 3;
                    obj.applyBoundaryConditions(d);
                end
            end
        end
        
        function set.array(obj,value)
            obj.pArray.array = value;
            for d = 1:3; % Do not try to force BCs in a nonfluxable direction
               if obj.gridSize(d) > 3; obj.applyBoundaryConditions(d); end
            end
        end
        
        %___________________________________________________________________________________________________ GS: gridSize
        % Specifies the grid size of the data array.
        function result = get.gridSize(obj)
            result = size(obj.pArray);
            if ( length(result) < 3), result = [result 1]; end
        end
        
        function set.gridSize(obj,value)
            warning('ImogenArray:GridSize',['The gridSize property is read-only property which is intrinsic' ...
                'to the stored array. It is %s and cannot be set to %s. It is an intrins'], mat2str(size(obj.pArray)), value);
        end
        
        %___________________________________________________________________________________________________ GS: idString
        function result = get.idString(obj)
            result = obj.id{1};
            if length(obj.id) > 1
                for i=2:length(obj.id)
                    result = strcat(result,'_',obj.id{i});
                end
            end
        end
        
        function set.idString(obj,value)
            warning('ImogenArray:IdString',['The idString property is read-only and cannot be'...
                'set to %s directly on %s.'],value, obj.id{:});
        end
        
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        %___________________________________________________________________________________________________ ImogenArray
        % Creates a new ImogenArray object according to the specified inputs.
        %>> id          Identification information for the new object.                      cell/str
        %>< run         Run manager object.                                                 ImogenManager
        %>> statics     Static arrays and values structure.                                 struct
        function obj = ImogenArray(component, id, manager, statics)
            if (nargin == 0 || isempty(id)); return; end
            if ~isa(id,'cell');    id = {id}; end
            obj.pArray = GPU_Type();
            obj.pShiftEnabled   = true(1,3);
            obj.component       = component;
            obj.id              = id;
            obj.pManager   = manager;
            
            manager.attachBoundaryConditions(obj);

            obj.boundaryData.rawStatics = statics; % Store this for when we have an array
            % to precompute stuff with.
            obj.boundaryData.compIndex = []; % Used to mark whether we're using statics
            obj.pBCUninitialized = true;
        end
        
        %___________________________________________________________________________________________________ cleanup
        % Cleans up the ImogenArray by emptying the data array, reducing memory requirements.
        function cleanup(obj)
            obj.pArray.clearArray();
        end
        
        function delete(obj)
	    obj.cleanup();
            if isfield(obj, 'boundaryData')
                obj.boundaryData.staticsData.clearArray();
            end
        end
        
        %___________________________________________________________________________________________________ shift
        % Shifts the input array along the specified direction and by the specified number of cells
        % according to the edgeshift function specified by the edgeshifts cell array.
        % * DIRECT        The direction along which to shift (1,2,3)                  int
        % * nCells        The number of cells to shift.                               int
        % # result        The shifted array.                                          double    (Nx,Ny,Nz)
        function result = shift(obj, DIRECT, nCells)
            upperLowerIndex = 1 + (nCells > 0);
            
            if obj.pShiftEnabled(upperLowerIndex, DIRECT)
                result = obj.edgeshifts{upperLowerIndex, DIRECT}(obj.pArray, DIRECT, nCells, obj);
            else
                result = obj.pArray;
            end
            
            obj.applyBoundaryConditions();
        end
        
        %___________________________________________________________________________________________________ transparentEdge
        % Returns the stored edge conditions used by transparent boundary conditions.
        function result = transparentEdge(obj,dim,upper)
            result = obj.edgeStore.getEdge(upper, dim, obj.pArray, obj.bcModes{1 + upper, dim});
        end
        
        %___________________________________________________________________________________________________ idAsString
        % Converts the ImogenArray object's identifier (id) property from a cell array to a string for
        % display or file writing purposes.
        function result = idAsString(obj)
            result = stringBuffer( ['[' class(obj) ']'] );
            for i=1:length(obj.id),     result.add(['.' obj.id{i}]); end
            result = result.string;
        end
        
        %___________________________________________________________________________________________________ arrayIndexExchange
        % Flips the array and all associated subarrays such that direction i and the x (stride-of-1) direction
        % exchange places. Updates the array, all subarrays, and the static indices.
        function arrayIndexExchange(obj, toex, type)
            if type == 1; % Does not flip subarrays, they can just be overwritten
                cudaArrayRotateB(obj.gputag, toex); obj.pArray.flushTag();
            end
        end
        
        %___________________________________________________________________________________________________ applyBoundaryConditions
        % Applies the static conditions for the ImogenArray to the data array. This method is called during
        % array assignment (set.array).
        function applyBoundaryConditions(obj, direction)
            cudaStatics(obj, 8, obj.pManager.parent.geometry, direction);
        end
        
        %___________________________________________________________________________________________________ setupBoundaries
        % Function merges the raw statics supplied when the ImogenArray was created with an initial data
        % array and compiled a linearly indexed list for use setting Boundary Conditions quickly at runtime.
        % This function should be called once at start-time.
        function setupBoundaries(obj)
            statics = obj.boundaryData.rawStatics;
            % Get "other" statics for this array, those not implied by boundary conditions
            [SI, SV, SC] = statics.staticsForVariable(obj.id{1}, obj.component, statics.CELLVAR);
            
            % Compile them into 6 arrays of index/value/coefficient, one for each possible axes arrangement.
            [SI, SV, SC] = staticsPrecompute(SI, SV, SC, statics.geometry.localDomainRez);
            
            % Collect boundary conditions for each of x, y and z fluxing; precompiled into 6 nice lists
            boundaryConds = Edges(obj.bcModes, obj.pArray, 0, statics.geometry);
            
            % Merge these lists together with the "other" statics,
            % And compile the whole thing into one triplet of 1D arrays, plus indexing offsets
            [compIndex, compValue, compCoeff, obj.boundaryData.compOffset] = ...
                staticsAssemble(SI, SV, SC, boundaryConds.boundaryStatics);
            
            obj.boundaryData.staticsData = GPU_Type([compIndex compValue compCoeff]);
            obj.boundaryData.bcModes = obj.bcModes;
            
            obj.pBCUninitialized = false;
        end
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        %______________________________________________________________________ initializeShiftingStates
        % Determines which grid dimensions have extent, i.e. are greater than 1 cell, and activates shifting
        % on only those dimensions to speed up the execution of 1D & 2D simulations.
        function initializeShiftingStates(obj)
            obj.pShiftEnabled = false(2,3);
            for i=1:ndims(obj.pArray)
                obj.pShiftEnabled(:, i) = size(obj.pArray, i);
            end
        end
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                       S T A T I C  [M]
        
        %________________________________________________________________________________________ zeroIt
        % Sets all values below the 1e-12 tolerance to zero on the input array.
        % inArray   Input array to be zeroed.                                                   double
        function result = zeroIt(inArray)
            result = inArray .* (inArray > 1e-12);
        end
        
    end%STATIC
    
end
