classdef ShuOsherTubeInitializer < Initializer
% This test is a hydrodynamic shocktube where the left and the right states are given as follows. 
% Left: (ρ=3.857143; Vx= 2.629369, P = 10.33333) Right: (ρ=1 + 0.2 sin(5 π x); Vx=0; P=1). Essentially 
% it is a Mach=3 shock interacting with a sine wave in density. 
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
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
            obj.pureHydro        = 1;
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.45;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.bcMode.x         = 'const';
            obj.bcMode.y         = 'circ';
            obj.bcMode.z         = 'circ';
            
            obj.operateOnInput(input, [1024 1 1]);

               obj.lambda        = 8;
            obj.mach             = 3;
            obj.waveAmplitude    = .2;

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
            statics             = []; % No statics used in this problem
            potentialField      = [];
            selfGravity         = [];

	    geo = obj.geomgr;
	    geo.makeBoxSize(1);

            X = geo.ndgridSetIJ('pos');

            % Initialize parallel vectors and logicals
            left = (X < 1/obj.lambda);
            right = (X >= 1/obj.lambda);

            %--- Set array values ---%
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            % Compute shockwave
            j                   = HDJumpSolver(obj.mach,0,obj.gamma);
            Vxl                 = j.v(1,1)-j.v(1,2);

            % Define structure for the left half of the grid
            ener(left)          = j.Pgas(2);         
            mass(left)          = j.rho(2);
            mom(1,left)         = Vxl*mass(left);

            % Define right-hand structures
            P(right)            = 1;
            mass(right)         = 1 + obj.waveAmplitude * sin(2*pi*X(right)*obj.lambda);
            mom(1,right)        = 0;
                        
            % Compute energy density array
            ener = ener/(obj.gamma - 1) ...
            + 0.5*squish(sum(mom.*mom,1))./mass...
            + 0.5*squish(sum(mag.*mag,1));

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
    end
        
end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
