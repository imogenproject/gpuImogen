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
        function self = FluidManager() 
            self.viscosity    = ArtificialViscosity();
            self.fluidName    = 'some_gas';
            % Set defaults
            self.checkCFL     = 1;
            self.isDust       = 0;
        end
        
       
        function attachBoundaryConditions(self, element)
            if ~isempty(self.parent)
                self.parent.bc.attachBoundaryConditions(element);
            else
                %warning('DANGER: No parent run associated with this FluidManager; Assuming I am being used for simpleminded debugging; Forging boundary conditions...');
                for a = 1:2; for b = 1:3; element.bcModes{a,b} = ENUM.BCMODE_CIRCULAR; end; end
                element.bcHaloShare = zeros(2,3);
            end
        end

        function attachFluid(self, holder, mass, ener, mom)
            self.DataHolder = holder;

            self.mass = mass;
            self.ener = ener;
            self.mom = mom;
        end

        function processFluidDetails(self, details)
            if isfield(details,'isDust');   self.isDust   = details.isDust;   end
            if isfield(details,'checkCFL'); self.checkCFL = details.checkCFL; end
        end
        
        function DEBUG_uploadData(self, rho, E, px, py, pz)
            % HACK HACK HACK this is a butchered copypasta from uploadDataArrays
            % that function should be abstracted so it can simply be called from here instead.
            self.MASS_THRESHOLD = 0;
            self.MINMASS        = 0;
            self.parent         = [];
            
            DH = GPU_Type(rho);
            DH.createSlabs(5);
            
            GM = GeometryManager(size(rho));
            SI = StaticsInitializer(GM);
            
            a = GPU_getslab(DH, 0);
            dens = FluidArray(ENUM.SCALAR, ENUM.MASS, a, self, SI);
            
            a = GPU_setslab(DH, 1, E);
            etotal = FluidArray(ENUM.SCALAR, ENUM.ENER, a, self, SI);
            
            momentum  = FluidArray.empty(3,0);
            
            a = GPU_setslab(DH, 2, px);
            momentum(1) = FluidArray(ENUM.VECTOR(1), ENUM.MOM, a, self, SI);
            a = GPU_setslab(DH, 3, py);
            momentum(2) = FluidArray(ENUM.VECTOR(2), ENUM.MOM, a, self, SI);
            a = GPU_setslab(DH, 4, pz);
            momentum(3) = FluidArray(ENUM.VECTOR(3), ENUM.MOM, a, self, SI);

            self.attachFluid(DH, dens, etotal, momentum);
        end

%___________________________________________________________________________________________________ initialize
        function initialize(self)
            for i = 1:numel(self)
                self(i).viscosity.preliminary();
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
        
