classdef FluidArray < ImogenArray
% The class used for fluid mechanics variables including mass, momentum, and energy. This class 
% provides the functionality to handle second order TVD, relaxed fluxing on a scalar array.
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                            P U B L I C [P]
        store;          % Half step storage.                                        StorageArray
        staticFluxes;   % Specifies static fluxing.                                 bool
        threshold;      % Value below which the thresholdArrau reads zero.
    end%PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        thresholdArray;     % Data array with values below threshold set to zero.   double
    end %DEPENDENT
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]

%___________________________________________________________________________________________________ GS: thresholdArray
% Returns the threshold array where values below the threshold property value are set to zero. If
% the threshold is set to zero, which it is by default, the threshold array and the main data array
% are identical. The threshold array is used for mass density when sourcing the graviational 
% potential, to avoid gravity acting on low mass cells, which causes unnecessarily high velocities.
        function result = get.thresholdArray(obj)
            result = obj.pArray;
            if (obj.threshold > 0)
                result = result .* (result > obj.threshold);
            end
        end
        
    end%GET/SET
    
%===================================================================================================
    methods %                                                                       P U B L I C  [M]
        
%___________________________________________________________________________________________________ FluidArray
% Creates a new FluidArray object.
% obj = FluidArray(component, id, array, run, statics)
        function obj = FluidArray(component, id, array, run, statics)
        
            obj = obj@ImogenArray(component, id, run, statics);
            if isempty(id); return; end

            obj.initializeDependentArrays(component, id, run, statics);

            if numel(array) > 0; obj.initialArray(squeeze(array)); end

            obj.isZero          = false;
%            obj.pUninitialized  = false;
%            obj.initializeShiftingStates(); % FIXME: can we get rid of this?
%            obj.initializeBoundingEdges();

%            obj.readFades(run);
           
%            obj.finalizeStatics(); % Put normal and static boundary conditions together and cast to GPU

%            obj.array = GPU_Type(squeeze(array));
            obj.indexGriddim = obj.gridSize;

            if strcmpi(id, ENUM.MASS)
                obj.threshold   = run.fluid.MASS_THRESHOLD;
                obj.pFadesValue = run.fluid.MINMASS;
            else
                obj.threshold = 0;
            end
            
        end

        function initialArray(obj, array)

            initialArray@ImogenArray(obj, array);

            obj.store.initialArray(array);
            obj.store.cleanup();

        end
        
%___________________________________________________________________________________________________ cleanup
% Cleans up the dependent arrays and objects stored inside the FluidArray. The FluidArray.array is
% not altered by this routine as FluidArray data is preserved throughout the run.
        function cleanup(obj)
            obj.store.cleanup();
        end
                
%___________________________________________________________________________________________________ dataClone
% Clones this FluidArray object by making a new one and supplying the clone with copied data from
% the existing FluidArray.
        function new = dataClone(obj)
            new             = FluidArray([],[],[],[],[]);
            new.component   = obj.component;
            new.edgeshifts  = obj.edgeshifts;
            new.array       = obj.array;
            new.id          = obj.id;
            new.bcInfinity  = obj.bcInfinity;
            new.bcModes     = obj.bcModes;
            new.initializeShiftingStates();
            new.initializeBoundingEdges();
        end
        
    end %METHODS
    
%===================================================================================================
    methods (Access = private) %                                                  P R I V A T E  [M]
        
%___________________________________________________________________________________________________ initializeDependentArrays
% Creates the dependent array objects for the fluxing routines.
        function initializeDependentArrays(obj, component, id, run, statics)
            obj.store    = StorageArray(component, id, run, statics);
        end
        
    end

    
    
end %CLASS
