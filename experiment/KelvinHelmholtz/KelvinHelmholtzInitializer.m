classdef KelvinHelmholtzInitializer < Initializer

% Creates initial conditions for a Kelvin-Helmholtz instability, where two anti-parallel shearing 
% flows are seeded with noise along the shearing barrier. The perturbation of the barrier causes
% the well known Kelvin-Helmholtz instabiliity phenomena to form. It should be noted that to 
% prevent a large shock wave from forming across the barrier the two regions, while having differing
% mass density values, have equal pressures. This makes sense given the usual observation of this
% phenomena in clouds, where such conditions exist.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        mach;
        waveHeight;
        numWave;
        randAmp;
        massRatio;      % ratio of (low mass)/(high mass) for the flow regions.     double
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ KelvinHelmholtzInitializer
        function obj = KelvinHelmholtzInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'KelHelm';
            obj.info             = 'Kelvin-Helmholtz instability test';
            obj.pureHydro        = true;
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.85;
            obj.iterMax          = 1500;
            obj.mach             = 0.25;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.waveHeight         = 0;
            obj.numWave                 = 10;
            obj.randAmp                 = .1;

            obj.bcMode.x = 'circ';
            obj.bcMode.y = 'circ';
            obj.bcMode.z = 'mirror';
            
            obj.massRatio        = 8;
            obj.operateOnInput(input, [512 512 1]);

        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]       
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [fluid, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        
            %--- Initialization ---%
            statics             = [];
            potentialField      = [];
            selfGravity         = [];
            geo                 = obj.geomgr;
            
            gdr = geo.globalDomainRez;
            if gdr(3) == 1; obj.bcMode.z = 'circ'; end

            geo.makeBoxSize([1 1 1]);

            % Set various variables
            speed               = speedFromMach(obj.mach, obj.gamma, 1, 1/(obj.gamma-1), 0); % Gives the speed of the fluid in both directions

            % Initialize Arrays
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            % Initialize parallelized vectors
            [X, Y, Z]           = geo.ndgridSetIJK('pos');

            % Define the wave contact in parallel
            topwave             = .33 + obj.waveHeight*sin(obj.numWave*2*pi*X);
            bottomwave          = .66 + obj.waveHeight*sin(obj.numWave*2*pi*X);
            heavy               = (Y > topwave) & (Y < bottomwave);

            % Define properties of the various regions. The high-density region in the middle moves in positive X, while the low-density regions move in negative X.
            mom(1,:,:,:)        = -speed;
            mom(1,heavy)        = speed;
            mass(heavy)         = obj.massRatio;

            % Give the grid a random velocity component in both X and Y of a size 'randamp', then multiply by the mass array to give momentum.
            mom(1,:,:,:)        = mom(1,:,:,:) + obj.randAmp*(2*rand(size(mom(1,:,:,:)))-1);
            mom(2,:,:,:)        = mom(2,:,:,:) + obj.randAmp*(2*rand(size(mom(2,:,:,:)))-1);
            mom(1,:,:,:)        = squish(mom(1,:,:,:)).*mass;
            mom(2,:,:,:)        = squish(mom(2,:,:,:)).*mass;

            % Calculate energy density array
            ener = (maxFinderND(mass).^obj.gamma)/(obj.gamma - 1) ...             % internal
            + 0.5*squish(sum(mom.*mom,1))./mass ...                             % kinetic
            + 0.5*squish(sum(mag.*mag,1));                                      % magnetic

            fluid = obj.rhoMomEtotToFluid(mass, mom, ener);
            end
        end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
