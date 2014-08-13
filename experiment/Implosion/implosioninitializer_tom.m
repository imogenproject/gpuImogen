classdef implosion_tomInitializer < Initializer
% Creates initial conditions for a Kelvin-Helmholtz instability, where two anti-parallel shearing 
% flows are seeded with noise along the shearing barrier. The perturbation of the barrier causes
% the well known Kelvin-Helmholtz instabiliity phenomena to form. It should be noted that to 
% prevent a large shock wave from forming across the barrier the two regions, while having differing
% mass density values, have equal pressures. This makes sense given the usual observation of this
% phenomena in clouds, where such conditions exist.
%
% Unique properties for this initializer:
%     direction      % enumerated orientation of the baseline flow.              str
%     massRatio      % ratio of (low mass)/(high mass) for the flow regions.     double
%     mach           % differential mach number between flow regions.            double
%     perturb        % specifies if flow should be perturbed.                    logical
    
        
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
        mach;           % differential mach number between flow regions.            double
        perturb;        % specifies if flow should be perturbed.                    logical
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
        function obj = implosion_tomInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'implo_tom';
            obj.info             = 'tom implosion test';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 1500;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.bcMode.x = 'const';
            obj.bcMode.y = 'const';
            obj.bcMode.z = 'const';
            
            obj.direction        = implosion_tomInitializer.X;
            obj.massRatio        = 8;
            obj.mach             = 0.25;
            obj.perturb          = true;
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

            indeces = cell(1,3);
            for i=1:3;  indeces{i} = 1:obj.grid(i); end
                     
            obj.dGrid = 1./obj.grid;
  
            mass    = ones(GIS.pMySize);
            mom     = zeros([3 GIS.pMySize]);
            mag     = zeros([3 GIS.pMySize]);
            speed   = speedFromMach(obj.mach, obj.gamma, 1, 1/(obj.gamma-1), 0);

            half    = ceil(obj.grid/2);
	    onethird = ceil(obj.grid/3);
	    twothird = ceil(obj.grid*2/3);
            fields  = {obj.X, obj.Y, obj.Z};
	    halves = true;
	    thirds = false;
            if halves
	        for i=1:3
                    if strcmpi(obj.direction,fields{i})
                        index = indeces;                        
                        if (i ~= 1) 
                            index{1} = half(1):obj.grid(1);
                        else
                            index{2} = half(2):obj.grid(2);
                        end
			mass(index{:}) = 1/obj.massRatio*mass(index{:});
                        mom(i,:,:,:)     = speed*mass;
                        mom(i,index{:}) = -squeeze(mom(i,index{:}));
                        obj.bcMode.(fields{i}) = 'const';
                    else 
		        obj.bcMode.(fields{i}) = 'const'; 
		% To return to solid boundaries on top and bottom, change this back to 'const'
                    end
                end
	    end
            if thirds
                for i=1:3
                    if strcmpi(obj.direction,fields{i})
                        index = indeces;
                        if (i == 1)
                            index{2} = onethird(2):twothird(2);
                        else
                            index{1} = onethird(1):twothird(1);
                        end
                        mass(index{:}) = 1/obj.massRatio*mass(index{:});
                        mom(i,:,:,:)     = speed*mass;
                        mom(i,index{:}) = -squeeze(mom(i,index{:}));
                        obj.bcMode.(fields{i}) = 'const';
                    else
                        obj.bcMode.(fields{i}) = 'const';
                % To return to solid boundaries on top and bottom, change this back to 'const'
                    end
                end
            end

           
	    if obj.perturb
		
	        runWaved = 0; % Whether to have a sinusoidal wave 
                numPerturb = 8; % How many wave peaks to have

		runRandomly = 1; % Whether to seed the grid with momentum instabilities
                RandSeedTop = .01; % The maximum amplitude of random fluctuations as a fraction of initial conditions
                RandSeedBottom = .5;	 

		maxSpeed = max(max(max(speed)));
		maxMass = max(max(max(mass)));	
	        maxDensity = maxFinderND(mass);
		
		if runWaved == 1
                    x1=linspace(-numPerturb*pi,numPerturb*pi,obj.grid(1));
                    x=linspace(1,obj.grid(1),obj.grid(1));
                    y=(cos(x1)+1)*obj.grid(2)*.005+obj.grid(2)/2;
                
		    for i=1:(max(obj.grid));
                        mass(ceil(x(i)),ceil(y(i)))=maxDensity;
		        mom(1,ceil(x(i)),ceil(y(i)),1)=(maxSpeed*maxMass);
                    end
                    for j=1:max(obj.grid);
                        for i=0:max(obj.grid)-2;
                            if mass(j,max(obj.grid)-i) == maxDensity;
                                mass(j,max(obj.grid)-i-1) = maxDensity;
		   	        mom(1,j,max(obj.grid)-i-1,1) = (maxSpeed*maxMass);
                            end
                        end
                    end
		end
		if runRandomly == 1

		    if RandSeedTop == RandSeedBottom
		    mom(1,:,:,1) = mom(1,:,:,1) + RandSeedTop*(mom(1,:,:,1).*(2*rand(size(mom(1,:,:,1)))-1));
		    mom(2,:,:,1) = mom(2,:,:,1) + RandSeedTop*(mom(2,:,:,1).*(2*rand(size(mom(2,:,:,1)))-1));
                    else
		    if halves
                    mom(1,:,1:half(1),1) = mom(1,:,1:half(1),1) + RandSeedTop*(mom(1,:,1:half(1),1).*(2*rand(size(mom(1,:,1:half(1),1)))-1));
                    mom(2,:,1:half(2),1) = mom(2,:,1:half(2),1) + RandSeedTop*(mom(2,:,1:half(2),1).*(2*rand(size(mom(2,:,1:half(2),1)))-1));

mom(1,:,half(1):obj.grid(1),1) = mom(1,:,half(1):obj.grid(1),1) + RandSeedBottom*(mom(1,:,half(1):obj.grid(1),1).*(2*rand(size(mom(1,:,half(1):obj.grid(1),1)))-1));
mom(2,:,half(2):obj.grid(2),1) = mom(2,:,half(2):obj.grid(2),1) + RandSeedBottom*(mom(2,:,half(2):obj.grid(2),1).*(2*rand(size(mom(2,:,half(2):obj.grid(2),1)))-1));
		    end
		    if thirds
			% Stuff goes here for seeding the thirds properly!
		    end
		    end
		end
            end

	ener = (maxFinderND(mass)^obj.gamma)/(obj.gamma - 1) ...     	% internal
        + 0.5*squeeze(sum(mom.*mom,1))./mass ...             		% kinetic
        + 0.5*squeeze(sum(mag.*mag,1));                      		% magnetic
        end
    end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
