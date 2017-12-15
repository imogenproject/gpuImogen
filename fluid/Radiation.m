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
        setStrength;
        solve;          % Handle to raditaion function for simulation.              @func

        pureHydro; % copy from run.pureHydro
        
        active;
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
            obj.active   = 0;
        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function readSubInitializer(obj, SI)
	    obj.type                 = SI.type;
	    obj.exponent             = SI.exponent;
	    obj.initialMaximum       = SI.initialMaximum;
	    obj.coolLength           = SI.coolLength;
	    obj.strengthMethod       = SI.strengthMethod;
	    obj.setStrength          = SI.setStrength;
        
        if strcmp(obj.type, ENUM.RADIATION_NONE) == 0; obj.active = 1; end
	end

        function initialize(obj, run, fluid, mag)
%___________________________________________________________________________________________________ 
% Initialize the radiation solver parameters.
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

            if strcmp(obj.type, ENUM.RADIATION_NONE)
                obj.strength = 0;
                return
            end
                       
            if strcmp(obj.strengthMethod, 'preset')
                obj.strength = obj.setStrength;
            end

            if strcmp(obj.strengthMethod, 'inimax')         
                unscaledRadiation = GPU_Type(cudaFreeRadiation(fluid.mass, fluid.mom(1), fluid.mom(2), ...
                    fluid.mom(3), fluid.ener, mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, fluid.gamma, obj.exponent, 1));
            
                KE  = 0.5*(mom(1).array.^2 + mom(2).array.^2 + mom(3).array.^2)./ mass.array;
                Bsq = 0.5*(mag(1).array.^2 + mag(2).array.^2 + mag(3).array.^2);
            
	        % compute the strength so as to radiate (initialMaximum) of Eint away after t=1
                obj.strength   = (ener.array - KE - Bsq) ./ unscaledRadiation.array;
                obj.strength   = obj.initialMaximum * mpi_max(max(obj.strength(:)));

                if mpi_amirank0(); fprintf('Radiation strength: %f\n', obj.strength); end
            end
            
            if strcmp(obj.strengthMethod,'coollen')
                forceStop_ParallelIncompatible();
                % FIXME: must have *global* psi(1,0,0) available.
                vpre = mom(1).array(1,1,1) / mass.array(1,1,1); % preshock vx
                
                ppost = .75*.5*vpre^2; % strong shock approximation
                rhopost = 4;
                obj.strength = (.25*vpre/obj.coolLength) * 4^(obj.exponent-2) * ppost^(1-obj.exponent);
                
                fprintf('Radiation strength: %f\n', obj.strength);
            end

	    obj.pureHydro = run.pureHydro;
        end
        
%___________________________________________________________________________________________________ noRadiationSolver
% Empty solver for non-radiating cases. 
        function result = noRadiationSolver(obj, fluid, mag, dTime)
            result = 0;
        end
        
%___________________________________________________________________________________________________ opticallyThinSolver
% Solver for free radiation.
% FIXME remove hard-coded T=1.05*T_0 temperature cutoff, yuck... 
        function result = opticallyThinSolver(obj, fluid, mag, dTime)
            cudaFreeRadiation(fluid, mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, [fluid.gamma obj.exponent obj.strength * dTime 1.05 obj.pureHydro]);
        end
        
    end%PUBLIC

    
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]        
    end%STATIC
    
end%CLASS
        
