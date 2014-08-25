classdef RichtmyerMeshkovInitializer < Initializer

% Heavy fluid at the bottom, light fluid on top. Have some non-uniform interface between them like a
% sinusoid, then launch a shock plane wave down into the interface. This shock will reflect off the
% interface non-uniformly, driving the heavy fluid into the light fluid like a jet.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        mach;
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
        function obj = RichtmyerMeshkovInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 7/5;
            obj.runCode          = 'RM';
            obj.info             = 'Richtmyer-Meshkov instability test';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 1500;
            obj.mach             = 0.66;
	    obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.bcMode.x = 'circ';
            obj.bcMode.y = 'const';
            obj.bcMode.z = 'mirror';
            
            obj.massRatio        = 5;
	    obj.pureHydro        = true;
            obj.operateOnInput(input);

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
            statics = [];
            potentialField = [];
            selfGravity = [];

            GIS = GlobalIndexSemantics();

	   % GIS.makeDimNotCircular(1);
	   % GIS.makeDimNotCircular(2);

            mass    = ones(GIS.pMySize);
            mom     = zeros([3 GIS.pMySize]);
            mag     = zeros([3 GIS.pMySize]);
            P	    = ones(GIS.pMySize);
	    theta = 0;
	    [result] = HDJumpSolver(obj.mach^(-1), theta, obj.gamma);
	    
	    runWaved = true;
            if runWaved

		InterfaceOffset = -4;
		numPerturb = 1;
		HeightModifier = .05;

		x1=linspace(-numPerturb*pi,numPerturb*pi,obj.grid(1));
                x=linspace(1,obj.grid(1),obj.grid(1));
                y=-(cos(x1)+InterfaceOffset)*obj.grid(2)*HeightModifier+obj.grid(2)/2;


                for i=1:(max(obj.grid));
                    mass(ceil(x(i)),ceil(y(i)))=obj.massRatio;
                    mom(2,ceil(x(i)),ceil(y(i)),1)=-result.v(1,2)*obj.massRatio;
                end
                for j=1:max(obj.grid);
                    for i=1:max(obj.grid)-1;
                        if mass(j,i) == obj.massRatio;
                            mass(j,i+1) = obj.massRatio;
			    mom(2,j,i+1,1) = -result.v(1,2)*obj.massRatio;
			end
                    end
                end
            end	

%  result.rho = [rho1 rho2];
%  result.v = [vx1 vx2; vy1 vy2];
%  result.B = [0 0; 0 0; 0 0]; % For compatibility w/MHDJumpSolver output
%  result.Pgas = [P1 P2];
%  result.Etot = [P1/(gamma-1) + T1, P2/(gamma-1) + T2];
%  result.theta = theta;
%  result.sonicMach = ms;
%  result.error = [rho2*vx2 - rho1*vx1, (rho1*vx1^2 + P1 - (rho2*vx2^2+P2)), vx1*(T1+gamma*P1/gm1) - vx2*(T2+gamma*P2/gm1)];


	% Create the shocked region 
	addshock = true;
	if addshock
	    shockmargin = 20;

            mom(2,:,1:(floor(min(y)-shockmargin)),1) = (result.v(1,1)-result.v(1,2))*result.rho(2);
            mass(:,1:(floor(min(y)-shockmargin)),:) = result.rho(2);
            P(:,1:(floor(min(y)-shockmargin)),:) = result.Pgas(2);
	end

	    ener = P/(obj.gamma - 1) ...     				% internal
	    + 0.5*squeeze(sum(mom.*mom,1))./mass ...             		% kinetic
            + 0.5*squeeze(sum(mag.*mag,1));                      		% magnetic
            end
	end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
