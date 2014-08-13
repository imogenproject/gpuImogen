classdef ImplosionInitializer < Initializer

% This test creates a triangular region of low-density fluid in the lower left corner that causes an 
% implosion. The problem is essentially the Sod shocktube in a 2D box, with the shock and rarefaction
% launched along the diagonal x = y. This test is exquisitely sensitive to the ability of the algorithm
% to maintain symmetry across the diagonal x = y. It also provides a measure of the rate of numerical
% diffusion of contact discontinuities.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        direction;      % enumerated orientation of the baseline flow.              str
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
        function obj = ImplosionInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'IMPLOSION';
            obj.info             = 'Implosion symmetry test';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 1500;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.bcMode.x = 'mirror';
            obj.bcMode.y = 'mirror';
            obj.bcMode.z = 'mirror';
            
            obj.direction        = ImplosionInitializer.X;
            obj.massRatio        = 8;
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
	    
	    corner = true;

	    if corner

		Pcorner = 0.14;
		mcorner = .125; % = obj.massRatio^(-1)

		entrywise = false;		

		if entrywise
		    x = linspace(1,obj.grid(1),obj.grid(1));
		    mass = mass*mcorner;
		    P = P*Pcorner;
		    for i = 1:length(x)
			for j = 1:length(x)
			    if i + j > obj.grid(1)/2
				P(i,obj.grid(1)-j+1,1) = 1;
				mass(i,obj.grid(1)-j+1,1) = 1;
			    end
			end
		    end
		else
		   P(1:obj.grid(1)/2,obj.grid(2)/2+1:obj.grid(2)) = triu(Pcorner*ones(size(P)/2)-1)+1;
                   mass(1:obj.grid(1)/2,obj.grid(2)/2+1:obj.grid(2)) = triu(mcorner*ones(size(mass)/2)-1)+1;
		end


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
