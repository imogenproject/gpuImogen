classdef MagnetArray < ImogenArray
% Array class for Magnetic field array components and related functionality.
    
%===================================================================================================
    properties (Constant = true) %                                              C O N S T A N T  [P]
        INDEX         = [2, 3; 1, 3; 1, 2];
        REVERSE_INDEX = [0, 1, 2; 1, 0 , 2; 1, 2, 0];
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        staticFluxes;   % Specifies static fluxing.                             logical
        fluxes;         % Fluxes for the magnetic field.                        FluxArray(3)
        stores;         % Half step storage.                                    StorageArray(2)
        velGrids;       % Grid-aligned velocity.                                InitializedArray(2)
        wMags;          % Auxiliary flux.                                       InitializedArray(2)
        cellMag;        % Cell-centered magnetic field.                         InitializedArray
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
        function obj = MagnetArray(component, id, array, run, statics)
            obj         = obj@ImogenArray(component, id, run, statics);

            obj.initializeArrays(component, id, run, statics);

            if numel(array) > 0; obj.initialArray(squeeze(array)); end

%            obj.initializeShiftingStates();
%            obj.initializeBoundingEdges();

%            obj.finalizeStatics(); % puts boundary and "internal" statics together and casts to GPU
        end

        function initialArray(obj, array)

            initialArray@ImogenArray(obj, array);
            
            obj.stores(1).initialArray(array);
            obj.stores(1).cleanup();
            obj.stores(2).initialArray(array);
            obj.stores(2).cleanup();

            obj.cellMag.initialArray(array); % Make the cell centered array build its statics
            obj.updateCellCentered();

        end

%___________________________________________________________________________________________________ flux
        function result = flux(obj, index)
            result = obj.fluxes(MagnetArray.REVERSE_INDEX(obj.component,index));
        end
        
%___________________________________________________________________________________________________ wMag
        function result = wMag(obj, index)
            result = obj.wMags(MagnetArray.REVERSE_INDEX(obj.component,index));
        end

%___________________________________________________________________________________________________ store
        function result = store(obj, index)
            result = obj.stores(MagnetArray.REVERSE_INDEX(obj.component,index));
        end

%___________________________________________________________________________________________________ velGrid
        function result = velGrid(obj, index)
            result = obj.velGrids(MagnetArray.REVERSE_INDEX(obj.component,index));
        end
        
%___________________________________________________________________________________________________ cleanup
        function cleanup(obj)
            for i=1:2
                obj.fluxes(i).cleanup();
                obj.stores(i).cleanup();
%                obj.wMags(i).cleanup();
                obj.velGrids(i).cleanup();
            end
        end        
        
    end%PUBLIC
    
%===================================================================================================
    methods (Access = protected) %                                            P R O T E C T E D  [M]
        
%___________________________________________________________________________________________________ initializeArrays
% Initializes all the secondary array objects owned by the MagnetArray object.
        function initializeArrays(obj, component, id, run, statics)
            obj.fluxes   	= FluxArray.empty(2,0);
            obj.stores  	= StorageArray.empty(2,0);
%            obj.wMags    	= InitializedArray.empty(2,0);
           obj.velGrids 	= InitializedArray.empty(2,0);
            for i=1:2
                comp = MagnetArray.INDEX(component,i);
                compID = ['mc_' num2str(comp)];
                obj.fluxes(i)	= FluxArray(component, {id, FluxArray.FLUX, compID}, run, statics);
                obj.stores(i) 	= StorageArray(component, {id, StorageArray.STORE, compID}, run, statics);
%                obj.wMags(i)    = InitializedArray(component, {id, 'wMag', compID}, run, statics);
                obj.velGrids(i) = InitializedArray(component, {id, 'velGrid', compID}, run, statics);
            end
            
            obj.cellMag = InitializedArray(component, id, run, statics);
        end
        
    end%PROTECTED
    
end %CLASS
