classdef ShuOsherTubeInitializer < Initializer
%This test is a very simple hydrodynamic shocktube filled with a γ=1.4 gas of uniform density 
% ρ = 1 and three different pressures. In the leftmost tenth the pressure is 1000, in the rightmost 
% tenth the pressure is 100, and in the center the pressure is a much smaller 0.01. The boundary 
% conditions are reflecting at both ends. Immediately both sides launch strong shocks towards the 
% center and strong rarefactions towards the walls. The various initial shocks, rarefactions, and 
% their reflections proceed to interact nonlinearly. 
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
   	lambda;
	mach;
	waveAmplitude;
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
        function obj = ShuOsherTubeInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'ShuOsher';
            obj.info             = 'Shu and Osher''s Shock tube test';
            obj.mode.fluid		 = true;
            obj.mode.magnet		 = false;
            obj.mode.gravity	 = false;
            obj.cfl				 = 0.45;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            
            obj.bcMode.x         = ENUM.BCMODE_CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 4, 4]);

   	    obj.lambda 		= 8;
	    obj.mach 		= 3;
	    obj.waveAmplitude 	= .2;

            obj.pureHydro = 1;
      
	end
        
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        
% This test is a hydrodynamic shocktube where the left and the right states are given as follows. 
% Left: (ρ=3.857143; Vx= 2.629369, P = 10.33333) Right: (ρ=1 + 0.2 sin(5 π x); Vx=0; P=1). Essentially 
% it is a Mach=3 shock interacting with a sine wave in density. 


            %--- Initialization ---%
            statics               = []; % No statics used in this problem
            potentialField        = [];
            selfGravity           = [];
	
	    GIS = GlobalIndexSemantics();
	    X = GIS.ndgridSetXYZ();

	    X = X/obj.grid(1);
	    left = (X < 1/obj.lambda);
	    right = (X >= 1/obj.lambda);

            %--- Set array values ---%
            mass                  = ones(GIS.pMySize);
            mom                   = zeros([3, GIS.pMySize]);
            mag                   = zeros([3, GIS.pMySize]);
            P                  	  = ones(GIS.pMySize)*.01;% / (obj.gamma - 1);


	    j 			  = HDJumpSolver(obj.mach,0,obj.gamma);

	    Vxl = j.v(1,1)-j.v(1,2);




	    P(left) 	  		  = j.Pgas(2); 	
	    mass(left)			  = j.rho(2);
	    mom(1,left)		 	  = Vxl*mass(left);

	    P(right)		 	  = 1;
	    mass(right)		  	  = 1 + obj.waveAmplitude * sin(2*pi*X(right)*obj.lambda);
	    mom(1,right)		  	  = 0;
			

ener = P/(obj.gamma - 1) ...
	+ 0.5*squeeze(sum(mom.*mom,1))./mass...
	+ 0.5*squeeze(sum(mag.*mag,1));





		obj.dGrid = 1/obj.grid(1);
            
    end
        
end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
