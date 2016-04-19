classdef MagnetArray < ImogenArray
% Array class for Magnetic field array components and related functionality.
    
%===================================================================================================
    properties (Constant = true) %                                              C O N S T A N T  [P]
        INDEX         = [2, 3; 1, 3; 1, 2];
        REVERSE_INDEX = [0, 1, 2; 1, 0 , 2; 1, 2, 0];
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        cellMag;        % Cell-centered magnetic field.                         ImogenArray
    end%PUBLIC

%===================================================================================================
    properties (SetAccess = private, GetAccess = private) %                       P R I V A T E  [P]   
    end
    
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]       
    end%GET/SET
    

%===================================================================================================
    methods %                                                                       P U B L I C  [M]
        
%___________________________________________________________________________________________________ updateCellCentered
% Updates the cell centered magnetic field object, which is used in the fluid fluxing routines.
        function updateCellCentered(obj)
            if size(obj.pArray,obj.component) > 1
                obj.cellMag.array = cudaFwdAverage(obj.pArray.GPU_MemPtr, obj.component);
            else
                obj.cellMag.array = obj.array; % without extent in the dimension, what is there to average or interpolate?
            end
        end

%___________________________________________________________________________________________________ MagnetArray
        function obj = MagnetArray(component, id, array, manager, statics)
            obj         = obj@ImogenArray(component, id, manager, statics);

            obj.initializeArrays(component, id, manager, statics);

            if numel(array) > 0; obj.initialArray(squish(array)); end
        end

        function initialArray(obj, array)
            initialArray@ImogenArray(obj, array);

            obj.cellMag.initialArray(array); % Make the cell centered array build its statics
            obj.updateCellCentered();
        end

%___________________________________________________________________________________________________ cleanup
        function cleanup(obj)
            obj.cellMag.cleanup();
        end        
        
    end%PUBLIC
    
%===================================================================================================
    methods (Access = protected) %                                            P R O T E C T E D  [M]
        
%___________________________________________________________________________________________________ initializeArrays
% Initializes all the secondary array objects owned by the MagnetArray object.
        function initializeArrays(obj, component, id, manager, statics)
            obj.cellMag = ImogenArray(component, id, manager, statics);
        end
        
    end%PROTECTED
    
end %CLASS
