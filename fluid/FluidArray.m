classdef FluidArray < ImogenArray
% The class used for fluid mechanics variables including mass, momentum, and energy. This class 
% provides the functionality to handle second order TVD, relaxed fluxing on a scalar array.
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                            P U B L I C [P]
        threshold;      % Value below which the thresholdArray reads zero.
    end%PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]

    end%GET/SET
    
%===================================================================================================
    methods %                                                                       P U B L I C  [M]
        
%___________________________________________________________________________________________________ FluidArray
% Creates a new FluidArray object.
% obj = FluidArray(component, id, array, run, statics)
        function obj = FluidArray(component, id, array, manager, statics, fakeIt)
        
            obj = obj@ImogenArray(component, id, manager, statics);
            if isempty(id); return; end

            if nargin == 6; if strcmp(fakeIt, 'fake')
                obj.storeOnCPU();
            end; end

            obj.initializeDependentArrays(component, id, manager, statics);

            if numel(array) > 0; obj.initialArray(squish(array)); end
            if obj.pStoredOnGPU
                obj.pArray.updateVectorComponent(component);
            end

            if strcmpi(id, ENUM.MASS)
                obj.threshold   = manager.MASS_THRESHOLD;
            else
                obj.threshold = 0;
            end
            
        end

        function initialArray(obj, array)
            initialArray@ImogenArray(obj, array);
        end
        
%___________________________________________________________________________________________________ cleanup
% Cleans up the dependent arrays and objects stored inside the FluidArray. The FluidArray.array is
% not altered by this routine as FluidArray data is preserved throughout the run.
        function cleanup(obj)

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
            new.bcModes     = obj.bcModes;
            new.initializeShiftingStates();
            new.initializeBoundingEdges();
        end
        
    end %METHODS
    
%===================================================================================================
    methods (Access = private) %                                                  P R I V A T E  [M]
        
%___________________________________________________________________________________________________ initializeDependentArrays
% Creates the dependent array objects for the fluxing routines.
        function initializeDependentArrays(obj, component, id, manager, statics)
        end
        
    end

    
    
end %CLASS
