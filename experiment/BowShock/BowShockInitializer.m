classdef BowShockInitializer < Initializer
    % Creates initial conditions for a bow shock simulation.
    %
    % Unique properties for this initializer:
    %   stencil      % File name for the statics stencil (must be in data dir).                 str
    %   staticType   % Enumerated specification of how to apply static values.                  str
    
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        PRIMARY_MODE = 'primary'; % Statics are applied to array classes.
        FLUX_MODE   = 'flux';    % Statics are applied to fluxes.
        FLUX_LR_MODE = 'fluxlr'; % Statics are applied to the left and right TVD fluxes only.
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        stencil;      % File name for the statics stencil (must be in data dir).            str
        staticType;   % Enumerated specification of how to apply static values.             str
        
        ballCells;    % 3x1, radii in all 3 dimensions                                      double
        ballCenter;   % 3x1, center of the ball                                             double
        
        magneticShock; % FIXME: class-ify this input once magnetism isn't broken
        magX;
        magY;
        
        ballLock;          % If true locks ball in place; If false, blast a clump of matter away
        % with a shockwave
        
        radBeta;          % Radiation rate = radBeta P^radTheta rho^(2-radTheta)
        radTheta; 
        radCoollen;
        
        ymirrorsym, zmirrorsym;
        
    end %PUBLIC
    
    %===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        % Density, pressure and mach of preshock flow
        preshockRho; preshockP; blastMach;
        
        % Surface state of ball (incl. outflow)
        ballRho; ballVr; ballXRadius; ballThermalPressure;
    end %DEPENDENT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pPreshockRho; pPreshockP; pBlastMach;
        pBallRho; pBallVr; pBallXRadius; pBallThermalPressure;
    end %PROTECTED
    
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
        %_______________________________________________________________________ BowShockInitializer
        function obj = BowShockInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'Bow';
            obj.info             = 'Bow shock trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.85;
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
            
            obj.staticType       = BowShockInitializer.PRIMARY_MODE;
            obj.stencil          = 'SmallSphere_800x256.mat';
            
            obj.ballCells        = [32 32 32];
            obj.ballCenter       = round(input/2);
            obj.pBallXRadius      = 1;
            obj.ballLock         = 1;

            obj.radCoollen = .3;
            
            obj.magX                = 0;
            obj.magY                = 0;
            
            obj.pPreshockRho         = 1;
            obj.pPreshockP           = 1;
            obj.pBlastMach           = 3;
            
            obj.pBallRho             = 1;
            obj.pBallVr              = 1;
            obj.pBallThermalPressure = .25;
            
            obj.radBeta = 1;
            obj.radTheta = 0.0;
            
            obj.ymirrorsym = 0;
            obj.zmirrorsym = 0;
            
            obj.operateOnInput(input, [800, 256, 1]);
        end
        
        function set.preshockP(self, P)
            P = P(1);
            self.pPreshockP = max(P, 1e-4);
        end
        function P = get.preshockP(self); P = self.pPreshockP; end
        
        function set.preshockRho(self, rho)
            rho = rho(1);
            self.pPreshockRho = max(rho, 1e-4);
        end
        function rho = get.preshockRho(obj); rho = obj.pPreshockRho; end
        
        function set.blastMach(self, M)
            M = M(1);
            self.pBlastMach = max(M, 0);
        end
        function M = get.blastMach(self); M = self.pBlastMach; end
        
        function set.ballRho(self, rho)
            rho = rho(1);
            self.pBallRho = max(rho, 1e-4);
        end
        function rho = get.ballRho(self); rho = self.pBallRho; end
        
        function set.ballVr(self, v)
            v = v(1);
            self.pBallVr = max(v, 1e-4);
        end
        function v = get.ballVr(self); v = self.pBallVr; end
        
        function set.ballXRadius(self, r)
            r = r(1);
            if r <= 0; error('Ball radius cannot be nonpositive'); end
            self.pBallXRadius = r;
        end
        function r = get.ballXRadius(self); r = self.pBallXRadius; end
        
        function set.ballThermalPressure(self, P)
            P = P(1);
            if P <= 0; error('Ball pressure cannot be nonpositive'); end
            self.pBallThermalPressure = P;
        end
        function P = get.ballThermalPressure(self); P = self.pBallThermalPressure; end
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
        %___________________________________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            % Returns the initial conditions for a bow shock simulation
            % USAGE: [mass, mom, ener, mag, statics, run] = getInitialConditions();
            potentialField = [];
            selfGravity = [];
           
            geo = obj.geomgr;

            Ltotal = obj.pBallXRadius * geo.globalDomainRez(1) / obj.ballCells(1);
            geo.makeBoxSize(Ltotal);
            %geo.makeBoxOriginCoord(ceil(geo.globalDomainRez/2));
            if obj.ymirrorsym
                obj.ballCenter(2) = 4.5;
                obj.bcMode.y = {ENUM.BCMODE_MIRROR, obj.bcMode.y};
            end
            if obj.zmirrorsym
                obj.ballCenter(3) = 4.5;
                obj.bcMode.z = {ENUM.BCMODE_MIRROR, obj.bcMode.z};
            end
            
            geo.makeBoxOriginCoord(ceil(obj.ballCenter));

            %--- Background Values ---%
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            momx = geo.zerosXYZ(geo.SCALAR);
            momy = geo.zerosXYZ(geo.SCALAR);
            momz = geo.zerosXYZ(geo.SCALAR);
            
            %--- Static Values ---%
            statics = StaticsInitializer(geo);
            
            [X, Y, Z] = geo.ndgridSetIJK('pos');
            Ledge = (X < (8-obj.ballCenter(1))/obj.ballCells(1)); % 8 leftmost cells - we establish plane flow here
            
            % The obstacle is an ellipsoid
            norm = sqrt(X.^2 + Y.^2 + Z.^2);
            ball = (norm <= 1.0);
            
            % Set minimum mass; Solve the hydro jump for the incoming blast
            % FIXME no no no, hardcoded parameters BAD
            obj.minMass = min(obj.pPreshockRho, obj.pBallRho) / 10000;
            
            blast = HDJumpSolver(obj.pBlastMach, 0, obj.gamma);
            xedge = max(round(obj.ballCenter(1) - obj.ballCells(1)-20), 16);
            preshockX = geo.toLocalIndices(1:xedge);
            postshockX = geo.toLocalIndices((xedge+1):geo.globalDomainRez(1));
            
            % Density distribution of background fluid
            mass(postshockX,:,:) = obj.pPreshockRho*blast.rho(2);
            mass(preshockX,:,:)  = obj.pPreshockRho;
            
            % Set the momentum of the incoming blast, if applicable
            if obj.pBlastMach > 1
                momx(postshockX,:,:) = obj.pPreshockRho*blast.rho(2)*( blast.v(1,2));
                momx(preshockX,:,:) = obj.pPreshockRho*blast.rho(1)*( blast.v(1,1));
            else
                momx(:,:,:) = obj.pPreshockRho*blast.rho(1)*(blast.v(1,1));
            end
            
            % Calculate the ball flow parameters
            ballRadii = (.9:.01:1.5) * obj.pBallXRadius;
            ballFlow = analyticalOutflow(obj.pBallXRadius, obj.pBallVr, obj.pBallRho, ...
                obj.pBallThermalPressure/obj.pBallRho^(5/3), ballRadii );
            
            % Interpolate mass & momentum onto the grid
            mass = interpScalarRadialToGrid(ballRadii, ballFlow.rho, [0 1.5*obj.pBallXRadius], X,Y,Z, mass);
            [momx, momy, momz] = interpVectorRadialToGrid(ballRadii, ballFlow.mom, [0 1.5*obj.pBallXRadius], ...
                X, Y, Z, momx, momy, momz);
            % "neatly" avoid the coordinate singularity at r=0
            momx(norm < .5) = 0;
            momy(norm < .5) = 0;
            momz(norm < .5) = 0;
            
            % We really do need to drop this silly 3xNxMxL thing at some point
            mom(1,:,:,:) = momx;
            mom(2,:,:,:) = momy;
            mom(3,:,:,:) = momz;
            
            % Calculate Etotal = P/(gamma-1) + KE for all points
            ener(postshockX,:,:) = obj.pPreshockP * blast.Pgas(2) /(obj.gamma-1);
            ener(preshockX,:,:)  = obj.pPreshockP / (obj.gamma-1);
            ener = interpScalarRadialToGrid(ballRadii, ballFlow.press / (obj.gamma-1), [0 1.5*obj.pBallXRadius], X,Y,Z,ener);
            ener = ener + .5*squish(sum(mom.^2,1))./mass;
            
            % Set up statics if we're locking the obstacle in place
            
            statics.indexSet{1} = indexSet_fromLogical(ball); % ball
            
            statics.valueSet = { mass(ball),  momx(ball), momy(ball), momz(ball), ener(ball) };
            
            clear momx; clear momy; clear momz;
            
            % Lock ball in place
            if obj.ballLock == true
                statics.associateStatics(ENUM.MASS, ENUM.SCALAR,    statics.CELLVAR, 1, 1);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(1), statics.CELLVAR, 1, 2);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(2), statics.CELLVAR, 1, 3);
                statics.associateStatics(ENUM.ENER, ENUM.SCALAR,    statics.CELLVAR, 1, 5);
                
                %if obj.mode.magnet == true;
                %    statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(1), statics.CELLVAR, 1, 10);
                %    statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(2), statics.CELLVAR, 1, 11);
                %end
                if geo.localDomainRez(3) > 1
                    statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(3), statics.CELLVAR, 1, 4);
                    %if obj.mode.magnet == true; statics.associateStatics(ENUM.MAG,  ENUM.VECTOR(3), statics.CELLVAR, 1, 1); end;
                end
                
            end
            
            if obj.radBeta > 0
                rad = RadiationSubInitializer();
                
                rad.type                      = ENUM.RADIATION_OPTICALLY_THIN;
                rad.exponent                  = obj.radTheta;
                
                rad.initialMaximum            = 1; % We do not use these, instead
                rad.coolLength                = obj.radCoollen; % We let the cooling function define dx
                rad.strengthMethod            = 'coollen';
                rad.setStrength               = obj.radBeta;
                
                obj.radiation = rad;
            end
            
            if (obj.magX == 0) && (obj.magY == 0); obj.pureHydro = 1; end

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
            
        end
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
