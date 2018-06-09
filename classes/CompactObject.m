classdef CompactObject < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]

    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]

    end %PROTECTED

    properties (SetAccess = protected, GetAccess = public)
        stateVector;
        history;
    end

%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function assignState(obj, newstate)
            obj.stateVector= newstate;
        end

        function incrementDelta(obj, delta)
            obj.stateVector = obj.stateVector + delta;
            obj.history = [obj.history; obj.stateVector];
        end

        function result   = status(obj)
            result.mass   = obj.stateVector(11);
            result.radius = obj.stateVector(4);
            result.pos    = obj.stateVector(1:3)';
            result.mom    = obj.stateVector(5:7)';
            result.L      = obj.stateVector(8:10)';
        end

    end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end %PROTECTED
                
%===================================================================================================        
    methods (Static = true) %                                                 S T A T I C    [M]

        function obj = CompactObject(ini)
            if nargin == 1
                if numel(ini) == 14
                    obj.stateVector = ini; else
                    % x y z r px py pz lx ly lz M rho_v rhog_v E_v
                    obj.stateVector = [0 0 0 1 0 0 0 0 0 0 1 1e-4 5e-4 1e-4];
                end
            else
                obj.stateVector = [0 0 0 1 0 0 0 0 0 0 1 1e-4 5e-4 1e-4];
            end
        end 

    end%PROTECTED
        
end%CLASS

