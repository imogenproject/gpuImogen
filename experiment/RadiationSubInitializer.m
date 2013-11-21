classdef RadiationSubInitializer < handle
% Sub initializer class for radiation variables.
	
%===================================================================================================
	properties (Constant = true, Transient = true) %							C O N S T A N T	 [P]
    end%CONSTANT
	
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %							P U B L I C  [P]
        type;           % Enumerated radiation type.                            string
        exponent;       % Radiation model exponent value.                       double
        initialMaximum; % Maximum percentage radiation loss compared to         double (%)
                        %   internal energy for initial conditions. 
        coolLength;
        setStrength;    % If strength method is 'preset' this simply defines the beta.
        strengthMethod;
    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %				   P R O T E C T E D [P]
    end %PROTECTED
	
	
	
	
	
%===================================================================================================
    methods %																	  G E T / S E T  [M]
	
%___________________________________________________________________________________________________ RadiationSubInitializer
        function obj = RadiationSubInitializer()
            obj.type            = ENUM.RADIATION_NONE;
            obj.exponent        = 0.5;
            obj.initialMaximum  = 0;
            obj.coolLength = 9999;
            obj.strengthMethod = 'inimax';
        end
        
    end%GET/SET
	
%===================================================================================================
    methods (Access = public) %														P U B L I C  [M]
	end%PUBLIC
	
%===================================================================================================	
	methods (Access = protected) %											P R O T E C T E D    [M]
	end%PROTECTED
		
%===================================================================================================	
	methods (Static = true) %													  S T A T I C    [M]
	end%PROTECTED
	
end%CLASS
