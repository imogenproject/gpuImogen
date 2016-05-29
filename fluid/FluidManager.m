classdef FluidManager < handle
% A FluidManager is a container for the grid state of a single fluid
% A simulation can have *multiple* fluids if running a multiphase flow simulation
% This canister holds the density, momentum, pressure functions for a given fluid
% 
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
        MINMASS;                  % Minimum allowed mass value.                         double
        MASS_THRESHOLD;           % Threshold above which gravity solver functions.     double
        thresholds;               % Threshold values for gravitational fluxing.         struct
        viscosity;                % Artificial viscosity object.                        ArtificialViscosity
        limiter;                  % Flux limiters to use for each flux direction.       cell(3)
    
        checkCFL;
        isDust; 
    end%PUBLIC
   
    properties (SetAccess = public, GetAccess = public)
        DataHolder; % A GPU_Type that holds the handle on the memory allocation
        mass, ener, mom; % Fluid state data                             ImogenArrays
    end
 
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                        P R I V A T E  [P]
        parent;               % Parent manager                                      ImogenManager

        fluidName;            % String describing which fluid this is               String
    end %PRIVATE
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
    end%GET/SET
    
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
%___________________________________________________________________________________________________ FluidManager
% Creates a new FluidManager instance.
        function obj = FluidManager() 
            obj.viscosity    = ArtificialViscosity();
            obj.fluidName    = 'some_gas';
            % Set defaults
            % Note that isDust being false will result in use of complex fluid riemann solver
            obj.checkCFL     = 1;
            obj.isDust       = 0;
        end

        function attachBoundaryConditions(obj, element)
            obj.parent.bc.attachBoundaryConditions(element);
        end

        function attachFluid(obj, holder, mass, ener, mom)
            obj.DataHolder = holder;

            obj.mass = mass;
            obj.ener = ener;
            obj.mom = mom;
        end

        function processFluidDetails(obj, details)
            if isfield(details,'isDust');   obj.isDust   = details.isDust;   end
            if isfield(details,'checkCFL'); obj.checkCFL = details.checkCFL; end
        end

%___________________________________________________________________________________________________ initialize
        function initialize(obj)
            for i = 1:numel(obj)
                obj(i).viscosity.preliminary();
            end
        end
        
        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = private) %                                                P R I V A T E    [M]
        
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
        
    end%STATIC
    
end%CLASS
        
