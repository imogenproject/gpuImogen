classdef RayleighTaylorInitializer < Initializer
% Run a simulation of the RT instability to test Imogen
%
%   useStatics        specifies if static conditions should be set for the run.       logical  
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    gravConstant;
    rhoTop;
    rhoBottom;
    P0;
    Bx;
    Bz;
    Kx;
    Ky;
    Kz;
    pertAmplitude;
    randomPert;

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]

    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]

    end %PROTECTED

%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ GravityTestInitializer
        function obj = RayleighTaylorInitializer(input)            
            obj                     = obj@Initializer();
            obj.grid = input;
            obj.runCode             = 'RAYLEIGH_TAYLOR';
            obj.info                = 'Rayleigh-Taylor instability test';
            obj.mode.fluid          = true;
            obj.pureHydro          = true;        
            obj.mode.magnet         = false;
            obj.mode.gravity        = false;
            obj.iterMax             = 300;
            obj.gamma               = 1.4;
            obj.bcMode.x            = 'circ';
            obj.bcMode.y            = 'mirror';
            obj.bcMode.z            = 'circ';

            obj.activeSlices.xy     = true;
            obj.timeUpdateMode      = ENUM.TIMEUPDATE_PER_STEP;
            obj.gravConstant        = 1;
            obj.gravity.constant    = .1;
            obj.gravity.solver      = ENUM.GRAV_SOLVER_EMPTY;

            obj.rhoTop              = 2;
            obj.rhoBottom           = 1;
            obj.P0                  = 2.5;
            obj.Bx                  = 0;
            obj.Bz                  = 0;

            obj.pertAmplitude = .0001;
            obj.Kx = 1;
            obj.Ky = 1;
            obj.Kz = 1;
            obj.randomPert = 1;
            
            obj.operateOnInput(input, [200, 100, 100]);
        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            potentialField = PotentialFieldInitializer();
            selfGravity = [];
            statics = [];

            geo = obj.geomgr;
            
            grid = geo.globalDomainRez;
            geo.makeBoxSize(grid(2)/grid(1));

            % Initialize Parallelized X,Y,Z vectors
            [X, Y, Z] = geo.ndgridSetIJK([0 0 0], obj.dGrid);

            % Define boundary
            Y0 = .5*obj.dGrid(2)*geo.localDomainRez(2);

            % Initialize Arrays
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            % Establish low density below, high density above
            mass(Y < Y0)     = obj.rhoBottom;
            mass(Y >= Y0)    = obj.rhoTop;

            % Establish variable to define pressure gradient
            if Y(1,1,1) < Y0
                Pnode              = Y(1,1,1)*obj.rhoBottom*obj.gravConstant;
            else
                Pnode              = obj.gravConstant*(Y0*obj.rhoBottom+(Y(1,1,1)-Y0)*obj.rhoTop);
            end

            obj.minMass = .0001*obj.rhoBottom;

            % Set gas pressure gradient to balance gravity
            ener = (obj.P0 - Pnode - obj.gravConstant * (cumsum(mass,2)-mass(1,1,1)) * obj.dGrid(2) );

            % Create gravity field
            potentialField.field = Y;
            potentialField.constant = obj.gravConstant;

        % If random perturbations are selected, it will impart random y-velocity to each column of the grid.
        % Otherwise, it will create a sinusoid of wavenumber Kx,Ky,Kz.
            if obj.randomPert == 0
                if geo.localDomainRez(3) == 1;
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5))/ 4;
                else
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5)) .* (1+cos(2*pi*obj.Kz*Z/.5))/ 8;
                end
            else
                w = (rand([geo.localDomainRez(1) geo.localDomainRez(3)])*-0.5) * obj.pertAmplitude;

                for y = 1:geo.localDomainRez(2); mom(2,:,y,:) = w; end
            end
            mom(2,:,:,:) = squish(mom(2,:,:,:)).*mass;

            % Don't perturb +y limit
	    if geo.edgeInterior(2,2) == 0; mom(2,:,(end-2):end,:) = 0; end
        
            % If doing magnetic R-T, turn on magnetic flux & set magnetic field & add magnetic energy
            if (obj.Bx ~= 0.0) || (obj.Bz ~= 0.0)
                obj.mode.magnet = true;
                mag(1,:,:,:) = obj.Bx;
                mag(3,:,:,:) = obj.Bz;
            end

            % Calculate Energy Density
            ener = ener/(obj.gamma - 1) ...
            + 0.5*squish(sum(mom.*mom,1))./mass...        
            + 0.5*squish(sum(mag.*mag,1));

            fluids = obj.stateToFluid(mass, mom, ener);
        end
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
