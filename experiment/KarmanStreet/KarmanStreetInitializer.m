classdef KarmanStreetInitializer < Initializer
% Description goes here

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
	mach;
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
        function obj = KarmanStreetInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'KARMAN';
            obj.info             = 'Karman Street test';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 1500;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.bcMode.x = {'bcstatic','const'};
            obj.bcMode.y = 'bcstatic';
            obj.bcMode.z = 'circ';
           
	    obj.mach = .5; %this is a test number
 
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
            statics = StaticsInitializer();
            potentialField = [];
            selfGravity = [];
            GIS = GlobalIndexSemantics();

	   % GIS.makeDimNotCircular(1);
	   % GIS.makeDimNotCircular(2);

            [X Y] = GIS.ndgridSetXY();
	    obj.dGrid = [1 1 1] / obj.grid(1);

	    % Initialize arrays
            mass    = ones(GIS.pMySize);
            mom     = zeros([3 GIS.pMySize]);
            mag     = zeros([3 GIS.pMySize]);
            P	    = ones(GIS.pMySize);

	    % Set various variables
	    radius = max(obj.grid)/50;
            speed   = speedFromMach(obj.mach, obj.gamma, 1, 1/(obj.gamma-1), 0);
	    mom(1,:,:,:) = speed*mass(:,:,:);

	    % Form the cylindrical obstruction
	    ball = ((X-.25*obj.grid(2)).^2 + (Y-obj.grid(2)/2).^2) < radius^2;

	    % Temporarily create syntactically clearer variables for statics
	    momx = squeeze(mom(1,:,:,:));
	    momy = squeeze(mom(2,:,:,:));
	    momz = squeeze(mom(3,:,:,:));
	   
	    % Set the momentum inside the obstruction to zero 
	    momx(ball) = 0;
	    
	    % Calculate energy density array
	    ener = P/(obj.gamma - 1) ...     		% internal
            + 0.5*squeeze(sum(mom.*mom,1))./mass ...    % kinetic
            + 0.5*squeeze(sum(mag.*mag,1));             % magnetic
          
	    % Make the obstruction static
	    statics.indexSet{1} = indexSet_fromLogical(ball); % ball
            statics.valueSet = { mass(ball),  momx(ball), momy(ball), momz(ball), ener(ball) };
            clear momx; clear momy; clear momz

            % Lock ball in place
                statics.associateStatics(ENUM.MASS, ENUM.SCALAR,    statics.CELLVAR, 1, 1);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(1), statics.CELLVAR, 1, 2);
                statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(2), statics.CELLVAR, 1, 3);
                statics.associateStatics(ENUM.ENER, ENUM.SCALAR,    statics.CELLVAR, 1, 5);
                if GIS.pMySize(3) > 1
                    statics.associateStatics(ENUM.MOM,  ENUM.VECTOR(3), statics.CELLVAR, 1, 4);
                end

        end
    end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
