classdef EinfeldtInitializer < Initializer
% These tests are Riemann problems where the left and right states are described for the first test
% (1-2-0-3) as follows. Left: (ρ=1; m=-2; n=0;e=3) Right: (ρ=1,m=2;n=0; e=3), where m = ρu ex and 
% n = ρv ey are the momentum densities, and u and v are the velocities in the x and y directions, 
% respectively. These initial conditions launch two rarefactions wave, one moving to the right and 
% one moving to the left. The other two initial conditions adopt this notation for the inital states
% as (1-1-0-5) and (1-1-2-5). This latter test introduces a shear flow perpendicular to the rarefactions. 
        
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
            obj.pureHydro        = 1;
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.bcMode.x         = ENUM.BCMODE_CONST;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            
            obj.operateOnInput(input, [1024, 4, 4]);

            obj.rhol               = 1;
            obj.ml                 = -2;
            obj.nl                 = 0;
            obj.el                 = 3;

            obj.rhor               = 1;
            obj.mr                 = 2;
            obj.nr                 = 0;
            obj.er                 = 3;
      
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
            statics               = []; % No statics used in this problem
            potentialField        = [];
            selfGravity           = [];
            GIS                   = GlobalIndexSemantics();
            GIS.setup(obj.grid);

            % Initialize Parallel Vectors        
            X                     = GIS.ndgridSetXY();
            obj.dGrid             = 1./obj.grid;
            half                  = ceil(obj.grid/2);
            left                  = (X < obj.grid(1)/2);
            right                 = (X >= obj.grid(1)/2);

            % Initialize Arrays
            [mass mom mag ener] = GIS.basicFluidXYZ();

            %--- Set Array Values ---%
            mass(left)            = obj.rhol;         %Density of left half
            mom(1,left)           = obj.ml;         %X momentum of left half
            mom(2,left)           = obj.nl;         %Y momentum of left half
            ener(left)            = obj.el;         %Total energy of left half
        
            mass(right)           = obj.rhor;        %Density of right half
            mom(1,right)          = obj.mr;         %X momentum of right half
            mom(2,right)          = obj.nr;         %Y momentum of right half
            ener(right)           = obj.er;        %Total energy of right half
    end
        
end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
