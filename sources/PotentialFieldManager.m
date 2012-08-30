classdef PotentialFieldManager < handle
% This is the management class for the potential solver. This is a singleton class to be accessed 
% using the getInstance() method and not instantiated directly. Currently the gravitational code is
% setup for a gravitational constant, G, of one.

    
%===================================================================================================
    properties (Constant = true, Transient = true) %                         C O N S T A N T     [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
    ACTIVE;             % Specifies that gravity spolver state                          logical 

    currentCoefficient  % Permits coefficient to be rescaled on the fly
    field;     % Adds a predefined fixed potential to any solved-for potential sparse
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

        function initialize(self, initialConds)

            if isempty(initialConds.field)
                self.ACTIVE      = false;
                self.currentCoefficient = 1;
                self.field = 0;
            else
                self.ACTIVE = true;
                self.currentCoefficient = initialConds.constant;
                self.field = GPU_Type(initialConds.field * self.currentCoefficient);
            end
        end

        
    end
 
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]
        
%___________________________________________________________________________________________________ GravityManager
% Creates a new GravityManager instance and intializes it with default settings.
        function self = PotentialFieldManager() 
            self.ACTIVE = false;
        end

    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                      S T A T I C    [M]
        
%___________________________________________________________________________________________________ getInstance
% Accesses the singleton instance of the GravityManager class, or creates one if none have
% been initialized yet.
        function singleObj = getInstance()
            persistent instance;
            if isempty(instance) || ~isvalid(instance) 
                instance = PotentialFieldManager();
            end
            singleObj = instance;
        end
        
    end%STATIC
    
end
