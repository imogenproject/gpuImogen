classdef SedovTaylorBlastWaveInitializer < Initializer
% Creates initial conditions for a Sedov-Taylor blast wave simulation. This is a high energy point
% located centrally in the grid that shocks outward in a spherical manner. A good test of how well
% the Cartesian grid handles non-aligned propagation. This problem is essentially the simulation of
% an explosion, and is developed based on the original nuclear simulation work of Sedov and Taylor.
%
% Unique properties for this initializer:
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        autoEndtime;
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
	pDepositRadius; % Radius (in cell) inside which energy will be deposited
	pBlastEnergy;
    end %PROTECTED
    
    
    
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ SedovTaylorBlastWaveInitializer
        function obj = SedovTaylorBlastWaveInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'ST_BLAST';
            obj.info             = 'Sedov-Taylor blast wave trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 10000;
            obj.bcMode.x         = ENUM.BCMODE_CONST;
            obj.bcMode.y         = ENUM.BCMODE_CONST;
            obj.bcMode.z         = ENUM.BCMODE_CONST;
            obj.activeSlices.xy  = true;
            obj.activeSlices.xyz = true;
            obj.ppSave.dim2      = 5;
            obj.ppSave.dim3      = 20;
            obj.pureHydro = true;

            obj.depositRadiusCells(3.5);
	    obj.setBlastEnergy(1);

            obj.autoEndtime = 1;

            obj.operateOnInput(input, [65, 65, 65]);
        end
               
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

	function depositRadiusCells(self, N); self.pDepositRadius = N; end
	function depositRadiusFraction(self, f); self.pDepositRadius = f*max(self.grid)/2; end
        function setBlastEnergy(self, Eblast);
	    self.pBlastEnergy = Eblast;
	 end


    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]

%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

	    GIS = GlobalIndexSemantics();
	    GIS.setup(obj.grid);

            %--- Initialization ---%
            statics         = [];
            potentialField  = [];
            selfGravity     = [];
            obj.dGrid       = 1./obj.grid;
            mass            = GIS.onesSetXYZ();
            mom             = zeros([3, GIS.pMySize]);
            mag             = zeros([3, GIS.pMySize]);
            
            if obj.autoEndtime
                disp('auto end time active: will run to blast radius = 0.45.');
                obj.timeMax = Sedov_timeToReachSize(obj.pBlastEnergy, .45, 1, obj.gamma, 2+1*(obj.grid(3)>1));
            end

            %--- Calculate Radial Distance ---%
	    [X Y Z] = GIS.ndgridSetXYZ(floor(obj.grid/2));
            distance  = sqrt(X.*X + Y.*Y + Z.*Z);
            clear X Y Z;

	    %--- Find the correct energy density Ec = Eblast / (# deposit cells) ---%
            % FIXME: Note that this is still 'wrong' in that it fails to integral average
            % over cells which only partially overlap the radius
            gv = floor(-obj.pDepositRadius-1):ceil(obj.pDepositRadius+1);
	    if obj.grid(3) == 1 % 2D
	        [xctr yctr] = ndgrid(gv,gv);
	        activeCells = (xctr.^2+yctr.^2 < obj.pDepositRadius^2);
	    else % 3D
		[xctr yctr zctr] = ndgrid(gv,gv,gv);
	        activeCells = (xctr.^2+yctr.^2+zctr.^2 < obj.pDepositRadius^2);
	    end
	
            nDepositCells = numel(find(activeCells));
            
            %--- Determine Energy Distribution ---%
	    ener            = 1e-8*mass/(obj.gamma-1); % Default to approximate zero pressure
	    ener(distance < obj.pDepositRadius) = obj.pBlastEnergy / (nDepositCells*prod(obj.dGrid));
        end
    
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
