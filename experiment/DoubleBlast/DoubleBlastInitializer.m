classdef DoubleBlastInitializer < Initializer
%This test is a very simple hydrodynamic shocktube filled with a gamma=1.4 gas of uniform density 
% rho = 1 and three different pressures. In the leftmost tenth the pressure is 1000, in the rightmost 
% tenth the pressure is 100, and in the center the pressure is a much smaller 0.01. The boundary 
% conditions are reflecting at both ends. Immediately both sides launch strong shocks towards the 
% center and strong rarefactions towards the walls. The various initial shocks, rarefactions, and 
% their reflections proceed to interact nonlinearly. 
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'X';
        Y = 'Y';
        Z = 'Z';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        pLeft;
        pRight;
        pMid;
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
        function obj = DoubleBlastInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 1.4;
            obj.runCode          = 'DoubleBlast';
            obj.info             = '1 Dimensional Double Blast Wave test';
            obj.mode.fluid       = true;
            obj.pureHydro        = 1;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.85;
            obj.iterMax          = 150;
            obj.ppSave.dim1      = 10;
            obj.ppSave.dim3      = 25;
            
            obj.bcMode.x         = 'mirror';
            obj.bcMode.y         = 'circ';
            obj.bcMode.z         = 'circ';

            obj.pLeft            = 1000;
            obj.pRight           = 100;
            obj.pMid             = .01;
            
            obj.operateOnInput(input, [1024, 1, 1]);      

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
            statics               = []; % No statics used in this problem
            potentialField        = [];
            selfGravity           = [];

            geo = obj.geomgr;
            
            geo.makeBoxSize(1); % double blast is done on box normalized to length = 1
            
            % Initialize parallel vectors
            Xpos = geo.ndgridSetIJ();
            left = (Xpos <= geo.globalDomainRez(1)/10);
            right= (Xpos > geo.globalDomainRez(1)*9/10);

            [mass, mom, mag, ~]   = obj.geomgr.basicFluidXYZ();

            P                        = obj.geomgr.onesXYZ() * obj.pMid;        %Pressure of inner zone = .01
            % Assign pressures of the two shockwave-creating regions
            P(left)  = obj.pLeft;
            P(right) = obj.pRight;

            % Compute energy density array
            ener = P/(obj.gamma - 1) ...
            + 0.5*squish(sum(mom.*mom,1))./mass...
            + 0.5*squish(sum(mag.*mag,1));

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
    end
        
end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
