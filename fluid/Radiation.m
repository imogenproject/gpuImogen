classdef Radiation < handle
% Contains the functionality for handling radiation based sources and sinks.
    
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
        type;           % Enumerated type of radiation model to use.                string
        strength;       % Strength coefficient for the radiation calculation.       double
        exponent;       % Radiation exponent.                                       double
        
        coolLength;     % Alternative cooling rate parameter
        initialMaximum; % Initial maximum radiation value used to calculate the     double
                        %   strength coefficient paramter.                          
        strengthMethod; 

        solve;          % Handle to raditaion function for simulation.              @func
    end%PUBLIC
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                        P R I V A T E  [P]
    end %PRIVATE
    
    
    
    
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ Radiation
% Creates a new Radiation instance.
        function obj = Radiation() 
            obj.strength = 1;
            obj.exponent = 0.5;
            obj.type     = ENUM.RADIATION_NONE;
        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

%___________________________________________________________________________________________________ preliminary
        function preliminary(obj)
            switch (obj.type)
                %-----------------------------------------------------------------------------------
                case ENUM.RADIATION_NONE
                    obj.solve = @obj.noRadiationSolver;
                %-----------------------------------------------------------------------------------
                case ENUM.RADIATION_OPTICALLY_THIN
                    obj.solve = @obj.opticallyThinSolver;
                %-----------------------------------------------------------------------------------
                otherwise
                    obj.type  = ENUM.RADIATION_NONE;
                    obj.solve = @obj.noRadiationSolver;
            end
        end

%___________________________________________________________________________________________________ 
% Initialize the radiation solver parameters.
        function initialize(obj, run, mass, mom, ener, mag)
            if strcmp(obj.type, ENUM.RADIATION_NONE)
                obj.strength = 0;
                return
            end
                       

            if strcmp(obj.strengthMethod, 'inimax')         
                unscaledRadiation = GPU_Type(cudaFreeRadiation(mass.gputag, mom(1).gputag, mom(2).gputag, mom(3).gputag, ener.gputag, mag(1).cellMag.gputag, mag(2).cellMag.gputag, mag(3).cellMag.gputag, run.GAMMA, obj.exponent, 1));
            
                kineticEnergy     = 0.5*(mom(1).array .* mom(1).array + mom(2).array .* mom(2).array ...
                                        + mom(3).array .* mom(3).array) ./ mass.array;
                magneticEnergy    = 0.5*(mag(1).array .* mag(1).array + mag(2).array .* mag(2).array ...
                                        + mag(3).array .* mag(3).array);
            
                obj.strength      = obj.initialMaximum* ...
                   minFinderND( (ener.array - kineticEnergy - magneticEnergy) ./ unscaledRadiation.array );

fprintf('Radiation strength: %f\n', obj.strength);
            end

            if strcmp(obj.strengthMethod,'coollen')
                vpre = mom(1).array(1,1) / mass.array(1,1); % preshock vx

                ppost = .75*.5*vpre^2; % strong shock approximation
                rhopost = 4;
                obj.strength = (.25*vpre/obj.coolLength) * 4^(obj.exponent-2) * ppost^(1-obj.exponent);

fprintf('Radiation strength: %f\n', obj.strength);
            end
        end
        
%___________________________________________________________________________________________________ noRadiationSolver
% Empty solver for non-radiating cases. 
        function result = noRadiationSolver(obj, run, mass, mom, ener, mag)
            result = 0;
        end
        
%___________________________________________________________________________________________________ opticallyThinSolver
% Solver for free radiation.
        function result = opticallyThinSolver(obj, run, mass, mom, ener, mag)
            cudaFreeRadiation(mass.gputag, mom(1).gputag, mom(2).gputag, mom(3).gputag, ener.gputag, mag(1).cellMag.gputag, mag(2).cellMag.gputag, mag(3).cellMag.gputag, run.GAMMA, obj.exponent, obj.strength * run.time.dTime);
        end
        
    end%PUBLIC
    
    
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]        
    end%STATIC
    
end%CLASS
        
