classdef SphericalShockInitializer < Initializer

% This test creates a spherical region of high-pressure fluid in the center of the grid that causes a 
% spherical shockwave to eminate outward, passing across the periodic boundaries to interact with
% itself, forming many small-scale structures such as RMIs and Kelvin-Helmholtz instabilities.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
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
        
%___________________________________________________________________________________________________ KelvinHelmholtzInitializer
        function obj = SphericalShockInitializer(input)
            obj = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'Sphere';
            obj.info             = 'Spherical Shock test';
            obj.pureHydro        = true;
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.4;
            obj.iterMax          = 1500;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 25;

            obj.bcMode.x = 'circ';
            obj.bcMode.y = 'circ';
            obj.bcMode.z = 'mirror';
            
            obj.operateOnInput(input, [256 256 256]);

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
            statics = [];
            potentialField = [];
            selfGravity = [];
            GIS = GlobalIndexSemantics();
            GIS.setup(obj.grid);

           % GIS.makeDimNotCircular(1);
           % GIS.makeDimNotCircular(2);

            % Initialize arrays
            [mass mom mag ener] = GIS.basicFluidXYZ();
            ener = .1*ener; % P = 0.1 by default

            % Define structure variables
            Psphere = 10;
            radius = obj.grid(1)/5;

            % Initialize parallel vectors and logicals
            [X Y Z] = GIS.ndgridSetXYZ();
            sphere = (sqrt((X-round(obj.grid(1)/2)).^2+(Y-round(obj.grid(2)/2)).^2)<=radius);
            ener(sphere) = Psphere;

            % Calculate energy density array
            ener = ener/(obj.gamma - 1) ...             % internal
            + 0.5*squish(sum(mom.*mom,1))./mass ...    % kinetic
            + 0.5*squish(sum(mag.*mag,1));             % magnetic
        end
    end%PROTECTED       
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
