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
   	    obj.pureHydro	    = true;	
            obj.mode.magnet         = false;
            obj.mode.gravity        = false;
            obj.iterMax             = 300;
            obj.gamma               = 1.4;
            obj.bcMode.x            = 'circ';
            obj.bcMode.y            = 'circ';
            obj.bcMode.z            = 'circ';

            obj.bcInfinity          = 4;

            obj.activeSlices.xy     = true;
            obj.timeUpdateMode      = ENUM.TIMEUPDATE_PER_STEP;
            obj.gravConstant 	    = 1;
            obj.gravity.constant    = .1;
            obj.gravity.solver      = ENUM.GRAV_SOLVER_EMPTY;

            obj.rhoTop              = 2;
            obj.rhoBottom           = 1;
            obj.P0                  = 2.5;
            obj.Bx                  = 0;
            obj.Bz		    = 0;

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
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            GIS = GlobalIndexSemantics();
            potentialField = PotentialFieldInitializer();
            selfGravity = [];
	    statics = [];

	    % Initialize Parallelized X,Y,Z vectors
            [X Y Z] = GIS.ndgridSetXYZ();
            obj.dGrid   = [.5 (.5*obj.grid(2)/obj.grid(1)) .5] ./ obj.grid;
            X = X * obj.dGrid(1);
            Y = Y * obj.dGrid(2);
            Z = Z * obj.dGrid(3);

	    % Define boundary
            Y0 = .5*obj.dGrid(2)*GIS.pMySize(2);

	    % Initialize Arrays
            mass             	= zeros(GIS.pMySize);
            mom         	= zeros([3 GIS.pMySize]);
	    P			= ones(GIS.pMySize);
            mag 		= zeros([3 GIS.pMySize]);

            % Establish low density below, high density above
            mass(Y < Y0)     = obj.rhoBottom;
            mass(Y >= Y0)    = obj.rhoTop;

	    % Establish variable to define pressure gradient
            if Y(1,1,1) < Y0
	        Pnode 	     = Y(1,1,1)*obj.rhoBottom*obj.gravConstant;
	    else
	        Pnode 	     = obj.gravConstant*(Y0*obj.rhoBottom+(Y(1,1,1)-Y0)*obj.rhoTop);
      	    end

            obj.minMass = .0001*obj.rhoBottom;

            % Set gas pressure gradient to balance gravity
            P = (obj.P0 - Pnode - obj.gravConstant * (cumsum(mass,2)-mass(1,1,1)) * obj.dGrid(2) );

	    % Create gravity field
	    potentialField.field = Y;
	    potentialField.constant = obj.gravConstant;

	% If random perturbations are selected, it will impart random y-velocity to each column of the grid.
	% Otherwise, it will create a sinusoid of wavenumber Kx,Ky,Kz.
            if obj.randomPert == 0
                if GIS.pMySize(3) == 1;
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5))/ 4;
                else
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5)) .* (1+cos(2*pi*obj.Kz*Z/.5))/ 8;
                end
            else
                w = (rand([GIS.pMySize(1) GIS.pMySize(3)])*-1) * obj.pertAmplitude;
                for y = 1:GIS.pMySize(2); mom(2,:,y,:) = w; end
            end
            mom(2,:,:,:) = squeeze(mom(2,:,:,:)).*mass;

        
            % If doing magnetic R-T, turn on magnetic flux & set magnetic field & add magnetic energy
            if (obj.Bx ~= 0.0) || (obj.Bz ~= 0.0)
                obj.mode.magnet = true;
                mag(1,:,:,:) = obj.Bx;
		mag(3,:,:,:) = obj.Bz;
            end

	    % Calculate Energy Density
	    ener = P/(obj.gamma - 1) ...
	    + 0.5*squeeze(sum(mom.*mom,1))./mass...	
            + 0.5*squeeze(sum(mag.*mag,1));
        end
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
