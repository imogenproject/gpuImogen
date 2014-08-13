classdef EinfeldtInitializer < Initializer
% These tests are Riemann problems where the left and right states are described for the first test
% (1-2-0-3) as follows. Left: (ρ=1; m=-2; n=0;e=3) Right: (ρ=1,m=2;n=0; e=3), where m = ρu ex and 
% n = ρv ey are the momentum densities, and u and v are the velocities in the x and y directions, 
% respectively. These initial conditions launch two rarefactions wave, one moving to the right and 
% one moving to the left. The other two initial conditions adopt this notation for the inital states
% as (1-1-0-5) and (1-1-2-5). This latter test introduces a shear flow perpendicular to the rarefactions. 
%
% Unique properties for this initializer:
%   direction       % Enumerated spatial orientation of the shock wave                      str
%   shockAngle      % Off-axis angle for the shockwave in degrees.                          double
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'X';
        Y = 'Y';
        Z = 'Z';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
	rhol;
	ml;
	nl;
	el;
	rhor;
	mr;
	nr;
	er;
        direction;      % Enumerated spatial orientation of the shock wave       	str
        shockAngle;     % Angle of the shock axis                                   double
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    
    
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ SodShockTubeInitializer
        function obj = EinfeldtInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'Einfeldt';
            obj.info             = 'Einfeldt Strong Rarefaction test';
            obj.mode.fluid		 = true;
            obj.mode.magnet		 = false;
            obj.mode.gravity	 = false;
            obj.cfl				 = 0.7;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.shockAngle       = 0;
            obj.direction        = 'EinfeldtInitializer.X';
            
            obj.bcMode.x         = ENUM.BCMODE_CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 4, 4]);

            obj.pureHydro = 1;

	    obj.rhol		= 1;
	    obj.ml		= -2;
	    obj.nl		= 0;
	    obj.el		= 3;

	    obj.rhor		= 1;
	    obj.mr		= 2;
	    obj.nr		= 0;
	    obj.er		= 3;
      
	end
        
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        
            %--- Initialization ---%
            obj.runCode           = [obj.runCode upper(obj.direction)];
            statics               = []; % No statics used in this problem
            potentialField        = [];
            selfGravity           = [];



% Left: (ρ=1; m=-2; n=0;e=3) 
% Right:(ρ=1; m=2;  n=0;e=3) 
% where m = ρu ex and n = ρv ey

% rho is mass density
% m = rho*u*ex, u is velocity vector of left half
% n = rho*v*ey, v is velocity vector of right half
% e = Total energy per unit area

            half                  = ceil(obj.grid/2);
            indices               = cell(1,3);
            for i=1:3
                indices{i} = 1:obj.grid(i);  
            end

            %--- Set array values ---%
            mass                  = ones(obj.grid);
            mom                   = ones([3, obj.grid]);
            mag                   = zeros([3, obj.grid]);
            ener                  = ones(obj.grid);% / (obj.gamma - 1);

	    mass(:,:,:) 	  = obj.rhol; 		%Density of left half
	    mom(1,:,:,:) 	  = obj.ml; 	%X momentum of left half
	    mom(2,:,:,:) 	  = obj.nl; 	%Y momentum of left half
	    ener(:,:,:) 	  = obj.el; 		%Total energy of left half
	
			

            mass(half(1):obj.grid(1),:,:)  = obj.rhor;		%Density of right half
	    mom(1,half(1):obj.grid(1),:,:) = obj.mr; 	%X momentum of right half
	    mom(2,half(1):obj.grid(1),:,:) = obj.nr; 	%Y momentum of right half
            ener(half(1):obj.grid(1),:,:)  = obj.er;		%Total energy of right half


            %--- Set shock array values according to flux direction ---%
            direct                = {'x', 'y', 'z'};
            i                     = find(strcmpi(obj.direction, direct), true);
            j                     = mod(i,2) + 1;
            
            for n=1:obj.grid(j)
                adjacentLen       = -half(j) + n - 1;
                lowerBound        = floor(half(i) ...
                                    - adjacentLen*tand(obj.shockAngle));
                lowerBound        = min(obj.grid(i), max(1, lowerBound));

                indices{i}        = lowerBound:obj.grid(i);
                indices{j}        = n;
            end
        
            %--- Adjust Cell Spacing ---%
            %       Problem is normalized so that the length from one end to the other of the shock
            %       tube is 1 unit length, no matter what the resolution. If the shock tube is 
            %       angled, then the normalization is scaled so that the end to end length of the 
            %       shock tube is unit length when cut down the center of tube along the shock
            %       normal.

            obj.dGrid             = 1./obj.grid;
            if obj.shockAngle > 0
                angle             = obj.shockAngle;
                criticalAngle     = atan(obj.grid(j)/obj.grid(i));
                if angle <= criticalAngle 
                    scale         = 1/(obj.grid(i)/cosd(angle));
                else
                    scale         = 1/(obj.grid(j)/sind(angle));
                end
                obj.dGrid(i)      = scale;
                obj.dGrid(j)      = scale;
            end

            %--- Determine the default slices to save ---%
            if ~obj.saveSlicesSpecified
                obj.activeSlices.(direct{i}) = true;
                
                if obj.shockAngle > 0
                    switch i+j
                        case 3     
                            obj.activeSlices.xy = true;
                        case 5
                            obj.activeSlices.yz = true;
                    end
                end
                
                obj.activeSlices.xyz = true;
            end
            
    end
        
end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
