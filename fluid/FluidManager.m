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

        radiation;                % Radiative emission properties for this fluid        Radiation
    
        checkCFL;
        isDust; 
        gamma;
    end%PUBLIC
   
    properties (SetAccess = public, GetAccess = public)
        DataHolder; % A GPU_Type that holds the handle on the memory allocation
        mass, ener, mom; % Fluid state data                             ImogenArrays
        parent;               % Parent manager                                      ImogenManager
    end
 
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                        P R I V A T E  [P]
        

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
            self.radiation    = Radiation();
            self.fluidName    = 'some_gas';
            % Set defaults
            self.checkCFL     = 1;
            self.isDust       = 0;
            self.gamma        = 5/3;
        end
      
        function setBoundaries(self, direction, what)
            if nargin < 3; what = [1 1 1 1 1 ]; else
                if numel(what) ~= 5; error('shit!'); end
            end

            if what(1); self.mass.applyBoundaryConditions(direction); end
            if what(2); self.ener.applyBoundaryConditions(direction); end
            if what(3); self.mom(1).applyBoundaryConditions(direction); end
            if what(4); self.mom(2).applyBoundaryConditions(direction); end
            if what(5); self.mom(3).applyBoundaryConditions(direction); end
        end

        function synchronizeHalos(self, direction, what)
            if nargin < 3; what = [1 1 1 1 1 ]; else
                if numel(what) ~= 5; error('shit!'); end
            end

            geo = self.parent.geometry;

            if what(1); cudaHaloExchange(self.mass,   direction, geo.topology, geo.edgeInterior(:,direction)); end
            if what(2); cudaHaloExchange(self.mom(1), direction, geo.topology, geo.edgeInterior(:,direction)); end
            if what(3); cudaHaloExchange(self.mom(2), direction, geo.topology, geo.edgeInterior(:,direction)); end
            if what(4); cudaHaloExchange(self.mom(3), direction, geo.topology, geo.edgeInterior(:,direction)); end
            if what(5); cudaHaloExchange(self.ener,   direction, geo.topology, geo.edgeInterior(:,direction)); end

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
            % Takes the { DataHolder, mass, ener, mom } objects returned in uploadFluidData()
            % and sets the FluidManager's like-named properties.
            self.DataHolder = holder;

            self.mass = mass;
            self.ener = ener;
            self.mom = mom;
        end

        function processFluidDetails(self, details)
            % This is called in uploadDataArrays with the ini.fluid(:).details structure
            % It sets all per-fluid properties, including adiabatic index and radiation
            % properties
            if isfield(details,'isDust');   self.isDust   = details.isDust;   end
            if isfield(details,'checkCFL'); self.checkCFL = details.checkCFL; end
            if isfield(details,'gamma');     self.gamma = details.gamma; end
            if isfield(details,'radiation');
                self.radiation.readSubInitializer(self, details.radiation);
            end
        end
        
        function DEBUG_uploadData(self, rho, E, px, py, pz)
            % DEBUG_uploadData(rho, E, px, py, pz) permits forging a working FluidManager
            % instance for test purposes, including the unitTest()
            % FIXME HACK HACK HACK this is a butchered copypasta from uploadDataArrays
            % FIXME that function should be abstracted so it can simply be called from here instead.
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

        function P = calcPressureOnCPU(self)
            P = (self.gamma - 1) * (self.ener.array - .5*(self.mom(1).array.^2+self.mom(2).array.^2+self.mom(3).array.^2)./self.mass.array);
        end

%___________________________________________________________________________________________________ initialize
        function initialize(self, mag)
            self.viscosity.preliminary();
            self.radiation.initialize(self.parent, self, mag);

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
        
