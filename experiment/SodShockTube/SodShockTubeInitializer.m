classdef SodShockTubeInitializer < Initializer
% Creates initial conditions for a Sod shock tube simulation. This is a canonical test of the 
% hydrodynamic portion of MHD codes as the solution can be determined analytically for comparison
% to metric the functionality of a code. Imogen has been tested thoroughly with this simulation and,
% as such, it is an excellent tool to verify the continued operation of the fluid routines.
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
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
	pShockNormal;    % Unitary 3-vector <Nx Ny Nz>. Points in this direction from the center
			 % form the low-density region
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ SodShockTubeInitializer
        function obj = SodShockTubeInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'SodST';
            obj.info             = 'Sod shock tube trial.';
            obj.mode.fluid		 = true;
            obj.mode.magnet		 = false;
            obj.mode.gravity	 = false;
            obj.cfl				 = 0.7;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.normal('X');
            
            obj.bcMode.x         = ENUM.BCMODE_CIRCULAR;%CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 4, 4]);

            obj.pureHydro = 1;
        end
        
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
	function normal(self, X, y, z)
	    if nargin == 3 % x, y, z scalar numbers passed separately
	        R = [X(1) y(1) z(1)];
		if norm(R) < 1e-10; R = [1 0 0]; end % Do not attempt to normalize too-small R
		R = R / norm(R);
	    else % Pick coordinate axis directions based only on X if one element, otherwise its a vector
		if numel(X) == 3;
		    if norm(X) < 1e-10; X = [1 0 0]; end
		    R = X / norm(X);
		else
			if isa(X, 'char')
			    switch X
				case 'X'; R = [1 0 0];
				case 'Y'; R = [0 1 0];
				case 'Z'; R = [0 0 1];
				default; warning('Default at invalid input','Given %s is not {X, Y or Z}: defaulting to X aligned shock', X); R = [1 0 0];
			    end
			else
			    switch X
				case 1; R = [1 0 0];
				case 2; R = [0 1 0];
				case 3; R = [0 0 1];
			 	default; warning('Default at invalid input','Given %g is not {1, 2, 3}: defaulting to X aligned shock', X); R = [1 0 0];
			    end
			end
		end

	    end

	self.pShockNormal = R;

	end

    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        
            %--- Initialization ---%
%            obj.runCode           = [obj.runCode upper(obj.direction)];
            statics               = []; % No statics used in this problem
            potentialField        = [];
            selfGravity           = [];
            half                  = floor(obj.grid/2);

	    GIS = GlobalIndexSemantics();
        GIS.setup(obj.grid);

	    %--- Compute the conditions for the domains ---%
	    [X Y Z] = GIS.ndgridSetXYZ(half + .5);
	    NdotX = (obj.pShockNormal(1)*X + obj.pShockNormal(2)*Y + obj.pShockNormal(3)*Z) > 0;

            %--- Set array values to high density condition ---%
            mass                  = ones(size(X));
            mom                   = zeros([3 size(X)]);
            mag                   = zeros([3 size(X)]);
            ener                  = mass/(obj.gamma - 1);

	    %--- Set low density condition ---%
	    mass(NdotX) = .125*mass(NdotX);
	    ener(NdotX) = .100*ener(NdotX);

            %--- Adjust Cell Spacing ---%
            %       Problem is normalized so that the length from one end to the other of the shock
            %       tube is 1 unit length, no matter what the resolution. If the shock tube is 
            %       angled, then the normalization is scaled so that the end to end length of the 
            %       shock tube is unit length when cut down the center of tube along the shock
            %       normal.
            obj.dGrid             = 1./obj.grid;
            % FIXME: This needs to account for angled shocks
            %--- Determine the default slices to save ---%
            if ~obj.saveSlicesSpecified
                obj.activeSlices.xyz = true;
            end
            
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
