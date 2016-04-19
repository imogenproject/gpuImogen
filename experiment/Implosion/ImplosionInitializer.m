classdef ImplosionInitializer < Initializer

% This test creates a triangular region of low-density fluid in the lower left corner that causes an 
% implosion. The problem is essentially the Sod shocktube in a 2D box, with the shock and rarefaction
% launched along the diagonal x = y. This test is exquisitely sensitive to the ability of the algorithm
% to maintain symmetry across the diagonal x = y. It also provides a measure of the rate of numerical
% diffusion of contact discontinuities.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
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
            GIS = GlobalIndexSemantics();
            GIS.setup(obj.grid);

           % GIS.makeDimNotCircular(1);
           % GIS.makeDimNotCircular(2);

            % Ensure that the grid is square.
            if obj.grid(1) ~= obj.grid(2)
                warning(sprintf('WARNING: grid [%g %g %g] was not square. grid(2) set to grid(1).\n', obj.grid(1), obj.grid(2), obj.grid(3)));
                obj.grid(2) = obj.grid(1);
            end
            obj.dGrid = 0.3 / obj.grid(1);

            % Initialize arrays
            [mass mom mag ener] = GIS.basicFluidXYZ();

            % Setup parallel vectors and structures
            [X Y] = GIS.ndgridSetXY();
            corn = (X + Y < obj.grid(1)/2);

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
