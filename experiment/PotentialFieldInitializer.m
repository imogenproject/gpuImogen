classdef PotentialFieldInitializer < handle
% Handles all of the various initialization properties for the gravitational functionality within
% Imogen.
        
%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        field;     % Contains a predefined, unchanging potential          double
        constant;           % Coupling constant to mass (G).                       double
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED


        
        
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ GravitySubInitializer
% Creates a new GravitySubInitializer object and sets the default settings.
        function obj = PotentialFieldInitializer()
            obj.constant         = 1;
            obj.field   = [];
        end
        
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
    end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                                                                          S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
