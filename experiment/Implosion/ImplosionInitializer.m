classdef ImplosionInitializer < Initializer

% This test creates a triangular region of low-density fluid in the lower left corner that causes an 
% implosion. The problem is essentially the Sod shocktube in a 2D box, with the shock and rarefaction
% launched along the diagonal x = y. This test is exquisitely sensitive to the ability of the algorithm
% to maintain symmetry across the diagonal x = y. It also provides a measure of the rate of numerical
% diffusion of contact discontinuities.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        Mcorner;
        Pcorner;
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
            obj.pureHydro        = true;
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
            
            obj.Mcorner = 0.125;
            obj.Pcorner = 0.14;

            obj.operateOnInput(input, [512 512 1]);

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
            statics = [];
            potentialField = [];
            selfGravity = [];
            
            geo = obj.geomgr;
            geo.makeBoxSize([1 1 1]);
            
            rez = geo.globalDomainRez;

            % Ensure that the grid is square.
            if rez(1) ~= rez(2)
                warning(sprintf('WARNING: grid [%g %g %g] was not square.\n', rez(1), rez(2), rez(3)));
            end
            
            % Initialize arrays
            [mass, mom, mag, ener] = geo.basicFluidXYZ();

            % Setup parallel vectors and structures
            [X, Y] = geo.ndgridSetIJ();
            corn = (X + Y < rez(1)/2);

            % Define the properties of the perturbed corner
            mass(corn) = obj.Mcorner;
            ener(corn)    = obj.Pcorner;

            % Calculate the energy density array
            ener = ener/(obj.gamma - 1) ...           % internal
            + 0.5*squish(sum(mom.*mom,1))./mass ...  % kinetic
            + 0.5*squish(sum(mag.*mag,1));           % magnetic

           fluids = obj.stateToFluid(mass, mom, ener);
        end
    end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
