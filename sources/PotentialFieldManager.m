classdef PotentialFieldManager < handle
% This is the management class for the potential solver. Currently the gravitational code is
% setup for a gravitational constant, G, of one.

    
%===================================================================================================
    properties (Constant = true, Transient = true) %                         C O N S T A N T     [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
    ACTIVE;             % Specifies whether static potential field is active                 logical 

    currentCoefficient  % Permits coefficient to be rescaled on the fly
    field;     % Adds a predefined fixed potential to any solved-for potential sparse

    starState; % [X Y Z, R, Px Py Pz, Lx Ly Lz, Mass, rhoVac EVac] 13x1
    end%PUBLIC
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                         P R I V A T E [P]
        parent;            % Manager parent                                             ImogenManger
    end %PROPERTIES 
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET    
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
%___________________________________________________________________________________________________ GravityManager
% Creates a new GravityManager instance and intializes it with default settings.
        function self = PotentialFieldManager() 
            self.ACTIVE = false;
        end

        function initialize(self, initialConds)
            run = self.parent;
            if isempty(initialConds.field)
                self.ACTIVE      = false;
                self.currentCoefficient = 1;
                self.field = 0;
            else
                self.ACTIVE = true;
                self.currentCoefficient = initialConds.constant;
                self.field = GPU_Type(initialConds.field * self.currentCoefficient);
                run.save.logPrint('Static potential field ACTIVE.\n');
            end
        end

        
    end
 
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C    [M]
        
    end%STATIC
    
end
