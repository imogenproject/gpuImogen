classdef BowShockInitializer < Initializer
% Creates initial conditions for a bow shock simulation. 
%
% Unique properties for this initializer:
%   stencil      % File name for the statics stencil (must be in data dir).                 str
%   staticType   % Enumerated specification of how to apply static values.                  str
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        PRIMAY_MODE = 'primary'; % Statics are applied to array classes.
        FLUX_MODE   = 'flux';    % Statics are applied to fluxes.
        FLUX_LR_MODE = 'fluxlr'; % Statics are applied to the left and right TVD fluxes only. 
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        stencil;      % File name for the statics stencil (must be in data dir).            str
        staticType;   % Enumerated specification of how to apply static values.             str

        ballCells;    % 3x1, radii in all 3 dimensions                                      double
        ballCenter;   % 3x1, center of the ball                                             double

        magneticShock;
	magX;
	magY;

        preshockRho;
        preshockP;
        blastMach;

        ballRho;           % Obstruction density, outflow velocity, radius and thermal pressure
        ballVr;            % Double scalar
        ballXRadius;
        ballThermalPressure;
        ballLock;          % If true locks ball in place; If false, blast a clump of matter away
                           % with a shockwave

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ BowShockInitializer
        function obj = BowShockInitializer(input)           
            obj = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'Bow';
            obj.info             = 'Bow shock trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.7;
            obj.iterMax          = 100;
            obj.bcMode.x         = 'const';
            obj.bcMode.y         = 'fade';
            if input(3) > 1
                obj.bcMode.z     = 'fade';
            else
                obj.bcMode.z     = 'circ';
            end
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;
            
            obj.staticType       = BowShockInitializer.PRIMAY_MODE;
            obj.stencil          = 'SmallSphere_800x256.mat';
           
            obj.ballCells        = [32 32 32];
            obj.ballCenter       = round(input/2);
            obj.ballXRadius      = 1;
            obj.ballLock         = 1;

            obj.magX                = 0;
            obj.magY                = 0;

            obj.preshockRho         = 1;
            obj.preshockP           = 1;
            obj.blastMach           = 3;

            obj.ballRho             = 1;
            obj.ballVr              = 1;
            obj.ballThermalPressure = .25;
         
            obj.operateOnInput(input, [800, 256, 1]);
        end
               
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        % Returns the initial conditions for a bow shock simulation
        % USAGE: [mass, mom, ener, mag, statics, run] = getInitialConditions();
            GIS = GlobalIndexSemantics();    
            potentialField = [];
            selfGravity = [];
            %obj.grind -> GIS.pMySize; this enables parallelization
            

            %--- Background Values ---%
            mass        = zeros(GIS.pMySize);
            mom         = zeros([3 GIS.pMySize]);
            mag         = zeros([3 GIS.pMySize]);
            ener        = zeros(GIS.pMySize);

            momx = zeros(GIS.pMySize);
            momy = zeros(GIS.pMySize);
            momz = zeros(GIS.pMySize);

            %--- Static Values ---%
            statics = StaticsInitializer();

            [X Y Z] = GIS.ndgridSetXYZ();
            Ledge = (X < 8); % Left edge - we establish plane flow here

            % The obstacle is a spheroid 
            X = (X - obj.ballCenter(1))/obj.ballCells(1);
            Y = (Y - obj.ballCenter(2))/obj.ballCells(2);
            Z = (Z - obj.ballCenter(3))/obj.ballCells(3);
            norm = sqrt(X.^2 + Y.^2 + Z.^2);
            ball = (norm <= 1.0);

            % Set minimum mass; Solve the hydro jump for the incoming blast
            obj.minMass = min(obj.preshockRho, obj.ballRho) / 1000;
            blast = HDJumpSolver(obj.blastMach, 0, obj.gamma);

            xedge = max(round(obj.ballCenter(1) - obj.ballCells(1)-20), 16);
            postshockX = GIS.toLocalCoords(1:xedge);
            preshockX = GIS.toLocalCoords((xedge+1):GIS.pMySize(1));

            % Density distribution of background fluid
            mass(postshockX,:,:) = obj.preshockRho*blast.rho(2);
            mass(preshockX,:,:)  = obj.preshockRho;

            % Set the momentum of the incoming blast, if applicable
            momx(postshockX,:,:) = obj.preshockRho*blast.rho(2)*(blast.v(1,1) - blast.v(1,2));

            % Calculate the ball flow parameters
            ballRadii = (.9:.01:1.5) * obj.ballXRadius;
            ballFlow = analyticalOutflow(obj.ballXRadius, obj.ballVr, obj.ballRho, ...
                                         obj.ballThermalPressure/obj.ballRho^(5/3), ballRadii );

            % Interpolate mass & momentum onto the grid
            mass = interpScalarRadialToGrid(ballRadii, ballFlow.rho, [0 1.5*obj.ballXRadius], X,Y,Z, mass);
            [momx, momy, momz] = interpVectorRadialToGrid(ballRadii, ballFlow.mom, [0 1.5*obj.ballXRadius], ...
                                                          X, Y, Z, momx, momy, momz);
            % "neatly" avoid the coordinate singularity at r=0
            momx(norm < .5) = 0;
            momy(norm < .5) = 0;
            momz(norm < .5) = 0;

            % We really do need to drop this silly 3xNxMxL thing at some point
            mom(1,:,:,:) = momx;
            mom(2,:,:,:) = momy;
            mom(3,:,:,:) = momz;
            
            % Calculate Etotal = P/(gamma-1) + T for all points
            ener(postshockX,:,:) = obj.preshockP * blast.Pgas(2) /(obj.gamma-1);
            ener(preshockX,:,:)  = obj.preshockP / (obj.gamma-1);
            ener = interpScalarRadialToGrid(ballRadii, ballFlow.press / (obj.gamma-1), [0 1.5*obj.ballXRadius], X,Y,Z,ener);
            ener = ener + .5*squeeze(sum(mom.^2,1))./mass;

            obj.dGrid = obj.ballXRadius / obj.ballCells(1);

            % Set up statics if we're locking the obstacle in place

            statics.indexSet{1} = indexSet_fromLogical(ball); % ball
            %statics.indexSet{2} = indexSet(GIS.pMySize, 1:2, 1:(GIS.pMySize(2)-0), 1:GIS.pMySize(3)); % incoming blast
            %statics.indexSet{3} = indexSet(GIS.pMySize, (GIS.pMySize(1)-4):(GIS.pMySize(1)-0), 1:GIS.pMySize(2), 1:GIS.pMySize(3)); % right edge
            %statics.indexSet{4} = indexSet(GIS.pMySize, 1:(GIS.pMySize(1)-2), 1:2, 1:GIS.pMySize(3)); % top edge

            statics.valueSet = { mass(ball),  momx(ball), momy(ball), momz(ball), ener(ball) };

            clear momx; clear momy; clear momz;

            % Lock ball in place
            if obj.ballLock == true
                statics.associateStatics(ENUM.MASS, ENUM.SCALAR,    statics.CELLVAR, 1, 1);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(1), statics.CELLVAR, 1, 2);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(2), statics.CELLVAR, 1, 3);
                statics.associateStatics(ENUM.ENER, ENUM.SCALAR,    statics.CELLVAR, 1, 5);

%            if obj.mode.magnet == true;
%                statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(1), statics.CELLVAR, 1, 10);
%                statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(2), statics.CELLVAR, 1, 11);
%            end
                if GIS.pMySize(3) > 1    
                    statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(3), statics.CELLVAR, 1, 4);
%                    if obj.mode.magnet == true; statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(3), statics.CELLVAR, 1, 1); end;
                end

            end

            if (obj.magX == 0) && (obj.magY == 0); obj.pureHydro = 1; end

        end

    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
