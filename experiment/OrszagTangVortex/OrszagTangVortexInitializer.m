classdef OrszagTangVortexInitializer < Initializer
% Creates initial conditions for an Orszag-Tang vortex test. This is THE canonical test for high
% fidelity magnetic fluxing, as it simulates the transition from sub-sonic to super-sonic mangetic
% flows for non-axis-aligned propagation vectors. There is no analytical solution to this problem,
% but it has been tested thoroughly by a number of different MHD codes to the point where a
% consensus has formed regarding its behavior. This particular problem has been setup to conform to
% the results of the Ryu paper on magnetohydrodynamics codes.
%
% Unique properties for this initializer:
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED 
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ OrszagTangVortexInitializer
        function obj = OrszagTangVortexInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'OTVortex';
            obj.info             = 'Orszag-Tang vortex trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = true;
            obj.mode.gravity     = false;
            obj.cfl              = 0.35;
            obj.iterMax          = 1500;
            obj.timeMax          = 0.48;
            obj.bcMode.x         = 'circ';
            obj.bcMode.y         = 'circ';
            obj.bcMode.z         = 'circ';
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;
            
            obj.operateOnInput(input, [256, 256, 1]);
        end
               
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%_____________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        
            %--- Initialization ---%
            statics         = StaticsInitializer();
            potentialField  = [];
            selfGravity     = [];

	    geo = obj.geomgr;
	    geo.makeBoxSize([1 1 1]);

            grid = GIS.globalDomainRez;
            mass            = 25/(36*pi)*GIS.onesSetXY();

            [x, y]          = GIS.ndgridSetIJ('coords'); 
            mom             = zeros([3 size(x,1) size(x,2) 1]);
            mom(1,:,:)      = - mass .* sin( 2*pi*y/(grid(2)-1) );
            mom(2,:,:)      =   mass .* sin( 2*pi*x/(grid(1)-1) );

            mag0            = 1/sqrt(4*pi);
            mag             = zeros([3 size(x,1) size(x,2) 1]);
            mag(1,:,:)      = - mag0*sin( 2*pi*y/(grid(2)-1) );
            mag(2,:,:)      =   mag0*sin( 4*pi*x/(grid(1)-1) );

            ener        = 5/(12*pi)/(obj.gamma - 1) ...                % internal
                            + 0.5*squish(sum(mom.*mom,1)) ./ mass ...  % kinetic
                            + 0.5*squish(sum(mag.*mag));               % magnetic
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
