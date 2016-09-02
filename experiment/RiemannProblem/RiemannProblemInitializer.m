classdef RiemannProblemInitializer < Initializer
    %
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        
        
    end %PUBLIC
    
    %===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pOctantRotation; % Define the rotation of the octants w.r.t the XYZ axes
        pOctantState;    % 8 cells representing the initial [rho vx vy vz P] state of each
        % octant of the fully general 3D Riemann Problem
        pCenterCoord;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
        %___________________________________________________________________________ SodShockTubeInitializer
        function obj = RiemannProblemInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'RP';
            obj.info             = 'Riemann Problem test.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.pOctantRotation = [0 0 0];
            obj.pOctantState = cell(8);
            obj.pCenterCoord = [.5 .5 .5];
            
            obj.bcMode.x         = ENUM.BCMODE_CIRCULAR;%CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 1, 1]);
            
            obj.pureHydro = 1;
        end
        
        
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        function orientate(self, yaw, pitch, roll)
            if nargin == 3; % All angular transforms given
                self.pOctantRotation = [yaw(1), pitch(1), roll(1)];
            else
                self.pOctantRotation = [0 0 0];
            end
            
        end
        
        function center(self, ctr)
            if numel(ctr) == 1; ctr = [1 1 1]*ctr; end
            self.pCenterCoord = ctr; % FIXME: check for mistakes in this
        end
        
        function demo_interactingRP1(self)
            % Sets up a 4-quadrant Riemann problem with a strongly interacting
            % central region. See
            % https://depts.washington.edu/clawpack/clawpack-4.3/applications/euler/2d/quadrants/www/
            a = 0.532258064516129;
            b = 1.206045378311055;
            c = 0.137992831541219;
            self.setupRiemann2D([1.5 0 0 0 1.5], ...
                                [a   b 0 0 0.3], ...
                                [a   0 b 0 0.3], ...
                                [c   b b 0 0.029032258064516]);
            self.gamma = 1.4;
            self.center([0.8 0.8 0.5]);
            self.orientate([0 0 0]);
            self.timeMax = 0.8;
        end

        function demo_interactingRP2(self)
        % Liska & Wendroff 2003, 2D RP case number 6
            self.setupRiemann2D([1 .75 -.5 0 1],[2,.75,.5,0,1],[3,-.75,-.5,0,1],[1,-.75,.5,0,1]);
            self.center([.5 .5 .5]);
            self.orientate([0 0 0]);
            self.timeMax = 0.3;
            self.gamma = 1.4;
        end
        
        function demo_SodTube(self)
            % Sets up the classic Sod tube problem
            self.center([0.5 0.5 0.5]);
            self.orientate(0,0,0);
            self.gamma = 1.4;
            self.timeMax = 0.2;

	    self.setupRiemann1D([1 0 0 0 1], [0.125 0 0 0 .1]);
        end

        function setupEinfeldt(self, mach, gam)
            self.center([0.5 0.5 0.5]);
            self.orientate(0,0,0);

            if (numel(mach) == 4) && (numel(gam) == 4)
	        % Use the Einfeldt [rho,m,n,e] formula to specify ICs
	        self.gamma = 1.4;
                el = mach; er = gam;

		left = [el(1) el(2) el(3) 0 0];
		right = [er(1) er(2) er(3) 0 0];
		% Calculate P
		left(5) = (el(4) - el(1)*(el(2)^2+el(3)^2)) / (self.gamma-1);
		right(5) = (er(4) - er(1)*(er(2)^2+er(3)^2)) / (self.gamma-1);
		if (left(5) < 0) || (right(5) < 0)
		    fprintf('Fatal problem: Einfeldt test initial conditions specify negative pressure.\n');
                    mpi_errortest(1);
		else
		    mpi_errortest(0);
		end
	    else
                self.gamma = gam;
                c0 = sqrt(gam);
                left = [1 -c0*mach(1) 0 0 1];
                right= [1  c0*mach(1) 0 0 1];
                if numel(mach) >= 2;
                    left(3) = mach(2)*c0;
                    right(3)= mach(2)*c0;
                end
                if numel(mach) >= 3
                    left(4) = mach(3)*c0;
                    right(4)= mach(3)*c0;
                end
	    end
        
	    self.setupRiemann1D(left, right);
        end

        function setupRiemann1D(self, Uleft, Uright)
            self.half(2, Uleft);
            self.half(1, Uright);
        end
        
        function setupRiemann2D(self, Ua, Ub, Uc, Ud)
            self.quadrant(1, Ua);
            self.quadrant(2, Ub);
            self.quadrant(3, Uc);
            self.quadrant(4, Ud);
        end
        
        function setupRiemann3D(self, Ua, Ub, Uc, Ud, Ue, Uf, Ug, Uh)
            self.octant(1, Ua);
            self.octant(2, Ub);
            self.octant(3, Uc);
            self.octant(4, Ud);
            self.octant(5, Ue);
            self.octant(6, Uf);
            self.octant(7, Ug);
            self.octant(8, Uh);
        end
        
        function half(self, n, state)
            switch n;
                case 1;
                    self.octant(1, state);
                    self.octant(3, state);
                    self.octant(5, state);
                    self.octant(7, state);
                case 2;
                    self.octant(2, state);
                    self.octant(4, state);
                    self.octant(6, state);
                    self.octant(8, state);
                otherwise;
                    warning('invalid half.');
            end
        end
        
        function quadrant(self, n, state)
            switch n;
                case 1;
                    self.octant(1, state);
                    self.octant(5, state);
                case 2;
                    self.octant(2, state);
                    self.octant(6, state);
                case 3;
                    self.octant(3, state);
                    self.octant(7, state);
                case 4;
                    self.octant(4, state);
                    self.octant(8, state);
                otherwise;
                    warning('Invalid quadrant.');
            end;
        end
        
        % OCTANT DEFINITIONS:
        % 1 +x+y+z (half 1, quadrant 1)
        % 2 -x+y+z (half 2, quadrant 2)
        % 3 +x-y+z (half 1, quadrant 3)
        % 4 -x-y+z (half 2, quadrant 4)
        % 5 +x+y-z (half 1, quadrant 1)
        % 6 -x+y-z (half 2, quadrant 2)
        % 7 +x-y-z (half 1, quadrant 3)
        % 8 -x-y-z (half 2, quadrant 4)
        function octant(self, n, state)
            if self.stateIsPhysical(state) && (n >= 1) && (n <= 8)
                self.pOctantState{n} = state;
            end
        end
        
        function tf = isOct(self, octant, x, y, z);
            switch octant;
                case 1; tf = (x >= 0) & (y >= 0) & (z >= 0);
                case 2; tf = (x <  0) & (y >= 0) & (z >= 0);
                case 3; tf = (x >= 0) & (y <  0) & (z >= 0);
                case 4; tf = (x <  0) & (y <  0) & (z >= 0);
                case 5; tf = (x >= 0) & (y >= 0) & (z <  0);
                case 6; tf = (x <  0) & (y >= 0) & (z <  0);
                case 7; tf = (x >= 0) & (y <  0) & (z <  0);
                case 8; tf = (x <  0) & (y <  0) & (z <  0);
            end
        end
        
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
        %___________________________________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            
            %--- Initialization ---%
            %            obj.runCode           = [obj.runCode upper(obj.direction)];
            statics               = [];
            potentialField        = [];
            selfGravity           = [];
            
            geo = obj.geomgr;
            if geo.pGeometryType == ENUM.GEOMETRY_SQUARE
                geo.makeBoxSize(1);
                geo.makeBoxOriginCoord(obj.pCenterCoord .* geo.globalDomainRez + 0.5);
            end
            
             % FIXME: need to do proper geometry calcs & switch cylindrical vs square stuff here... 
            %--- Compute the conditions for the domains ---%
            [X, Y, Z] = geo.ndgridSetIJK('pos');
            
            if geo.pGeometryType == ENUM.GEOMETRY_CYLINDRICAL
               midrad = (geo.affine(1)*2 + geo.d3h(1)*geo.globalDomainRez(1)) / 2; 
               X = X - midrad;
               Y = Y - pi;
            end
            
            r = obj.pOctantRotation(3);
            p = obj.pOctantRotation(2);
            y = obj.pOctantRotation(1);
            
            % FIXME: I think matlab has some things to do this for us...
            m = eye(3); % projection matrix;
            m = [1 0 0; 0 cos(r) sin(r); 0 -sin(r) cos(r)]*m; % roll about x
            m = [cos(p) 0 sin(p); 0 1 0; -sin(p) 0 cos(p)]*m; % pitch about y
            m = [cos(y) sin(y) 0; -sin(y) cos(y) 0; 0 0 1]*m; % yaw about z
            
            U = m(1,1)*X + m(2,1)*Y + m(3,1)*Z;
            V = m(1,2)*X + m(2,2)*Y + m(3,2)*Z;
            W = m(1,3)*Z + m(2,3)*Y + m(3,3)*Z;
            clear X Y Z;
            
            
            [mass, mom, mag, ener] = geo.basicFluidXYZ();
            px = 1.0*mass;
            py = 1.0*mass;
            pz = 1.0*mass;
            for oh = 1:8;
                psi = obj.pOctantState{oh};
                
                t = obj.isOct(oh, U, V, W);
                mass(t) = psi(1);
                px(t)   = psi(1)*psi(2);
                py(t)   = psi(1)*psi(3);
                pz(t)   = psi(1)*psi(4);
                ener(t) = .5*psi(1)*(psi(2)^2+psi(3)^2+psi(4)^2) + psi(5)/(obj.gamma-1);
            end
            
            mom(1,:,:,:) = px; clear px;
            mom(2,:,:,:) = py; clear py;
            mom(3,:,:,:) = pz; clear pz;
            
            if ~obj.saveSlicesSpecified
                obj.activeSlices.xyz = true;
            end
            
            fluids = obj.stateToFluid(mass, mom, ener);
            
        end
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                     S T A T I C    [M]
        
        function tf = stateIsPhysical(state)
            
            if numel(state) ~= 5; tf = false; else
                tf = true;
                if state(1) <= 0; tf = false; end;
                if state(5) <= 0; tf = false; end;
            end
            
        end
        
    end
end%CLASS
