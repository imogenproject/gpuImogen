classdef ShockBubbleInitializer < Initializer
% Creates imogen input data arrays for the fluid, magnetic-fluid, and gravity jet tests.
%
% Unique properties for this initializer:

        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        bubbleRadius;
        bubbleDensity;
        shockMach;

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ JetInitializer
        function obj = ShockBubbleInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'Bubble';
            obj.info             = 'Shock/Bubble collision trial';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.85;
            obj.iterMax          = 250;
            obj.bcMode.x         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.y         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.z         = ENUM.BCMODE_CONSTANT;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 10;
            obj.pureHydro = 1;
            
            obj.bubbleRadius = .1;
            obj.bubbleDensity = 10;
            
            obj.shockMach = 5;
            
            obj.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');
            
            obj.operateOnInput(input, [512 256 1]);
        end
 
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            potentialField = [];
            selfGravity = [];
            statics = [];
            
            geo = obj.geomgr;
            rez = geo.globalDomainRez;
            
            % Box height = 1, width = aspect ratio
            obj.geomgr.makeBoxSize(rez(1)/rez(2));
            obj.geomgr.makeBoxOriginCoord([rez(1)*(1.5*obj.bubbleRadius), rez(2)/2, rez(3)/2]);

            critx = 3*obj.bubbleRadius;
            
            [mass, vel, mag, eint] = obj.geomgr.basicFluidXYZ();

            [xp, yp, zp] = obj.geomgr.ndgridSetIJK('pos');
            
            shocker = HDJumpSolver(obj.shockMach, 0, obj.gamma);
            
            mass(xp < critx) = 1;
            mass(xp > critx) = shocker.rho(2);
            
            q = ones(size(xp));
            q(xp < critx) = shocker.v(1,1);
            q(xp >= critx)= shocker.v(1,2);
            
            vel(1,:,:,:) = q;
            
            eint(xp < critx) = shocker.Pgas(1)/(obj.gamma-1);
            eint(xp > critx) = shocker.Pgas(2)/(obj.gamma-1);

            if size(zp,3) > 1
                bubb = sqrt(xp.^2+yp.^2+zp.^2);
            else
                bubb = sqrt(xp.^2+yp.^2);
            end
            mass(bubb < obj.bubbleRadius) = obj.bubbleDensity;

            fluids = obj.rhoVelEintToFluid(mass, vel, eint);
        end
        
%___________________________________________________________________________________________________ toInfo
        function result = toInfo(obj)
            skips = {'X', 'Y', 'Z'};
            result = toInfo@Initializer(obj, skips);
        end                    
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
