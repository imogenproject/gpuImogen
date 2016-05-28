classdef NohTubeInitializer < Initializer
    %===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        v0;    % Initial implosion speed
	rho0;  % Initial density
	r0;    % Radius of implosion

	M0;    % Mach (large for analytical soln to work)
	
	t0;    % Positive: center is postshock, negative: center is vacuum
    end %PUBLIC
    
    %===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
        %___________________________________________________________________________ SodShockTubeInitializer
        function obj = NohTubeInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'NT';
            obj.info             = 'Noh tube implosion.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.bcMode.x         = ENUM.BCMODE_CIRCULAR;%CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 1, 1]);
            
            obj.pureHydro = 1;
        end
        
        
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
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
            
            GIS = GlobalIndexSemantics();
            GIS.setup(obj.grid);
            
	    obj.dGrid = [1 1 1]/obj.grid(1);
            half = floor(obj.grid/2);

            %--- Compute the conditions for the domains ---%
            [X Y Z] = GIS.ndgridSetXYZ(half + .5, obj.dGrid);
       
            R = sqrt(X.^2+Y.^2+Z.^2);

            spaceDim = 1;
            if obj.grid(2) > 1; spaceDim = spaceDim + 1; end
            if obj.grid(3) > 1; spaceDim = spaceDim + 1; end

            [mass mom mag ener] = GIS.basicFluidXYZ();

	    if obj.t0 > 0; % Begin with nonvacuum at r < r0
		% Solve the actual time associated with r0
		Dee = (1-obj.gamma)*obj.v0/2;
		t = obj.r0 / Dee;

		mass(R > obj.r0) = obj.rho0*(1 - obj.v0*t./R(R > obj.r0)).^(spaceDim-1);

		px = (X./R).*mass*obj.v0;;
		if spaceDim >= 2; py = (Y./R).*mass*obj.v0; end
		if spaceDim >= 3; pz = (Z./R).*mass*obj.v0; end

		px(R < obj.r0) = 0;
		py(R < obj.r0) = 0;
		pz(R < obj.r0) = 0;

		% KE plus small qty for finite pressure requirement, defined thru Mach
		% Define density at shock
		jc = (obj.gamma+1)/(obj.gamma-1);
		rhopre = obj.rho0*jc^(spaceDim-1);
		% The pressure for the Mach to be such
		p0 = rhopre*(obj.v0/obj.M0)^2/(obj.gamma*(obj.gamma-1));

		% Add this to preshock energy density assuming adiabaticity
		ener(R > obj.r0) = .5*obj.v0^2*mass(R > obj.r0) + p0*(mass(R > obj.r0)./rhopre).^obj.gamma / (obj.gamma-1);

		% Compute the central region
		% Shocked, stationary central region
		mcent = obj.rho0 * jc^spaceDim;
		mass(R <= obj.r0)= mcent;
		% mom is zero by default: yay
		ener(R <= obj.r0)= mcent*obj.v0^2 / 2;
	    else
		error('Do not support imploding shell yet: Set run.t0 > 0 please.\n');
        end
           
        mom(1,:,:,:) = px;
        if(spaceDim >= 2); mom(2,:,:,:) = py; end
        if(spaceDim >= 3); mom(3,:,:,:) = pz; end
 
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
