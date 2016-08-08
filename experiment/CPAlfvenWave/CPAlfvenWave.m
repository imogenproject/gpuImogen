classdef CPAlfvenWave < Initializer
% Creates initial conditions for the corrugation instability shock wave problem. The fundamental 
% conditions are two separate regions. On one side (1:midpoint) is in inflow region of accreting
% matter that is low mass density with high momentum. On the other side is a high mass density
% and low momentum region representing the start. The shockwave is, therefore, the surface of the
% star. This problem assumes a polytropic equation of state where the polytropic constant, K, is
% assumed to be 1 on both sides of the shock.
%
% Unique properties for this initializer:
%   perturbationType    enumerated type of perturbation used to seed.                   str
%   seedAmplitude       maximum amplitude of the seed noise values.                     double
%   theta               Angle between pre and post shock flows.                         double
%   sonicMach           Mach value for the preshock region.                             double
%   alfvenMach          Magnetic mach for the preshock region.                          double
        
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
        
%___________________________________________________________________________________________________ CorrugationShockInitializer
% Creates an Iiitializer for corrugation shock simulations. Takes a single input argument that is
% either the size of the grid, e.g. [300, 6, 6], or the full path to an existing results data file
% to use in loading
        function obj = CPAlfvenWave(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'CPAW';
            obj.info             = 'Nonlinear Alfven Wave Test';
            obj.mode.fluid       = true;
            obj.mode.magnet      = true;
            obj.mode.gravity         = false;
            obj.treadmill        = false;
            obj.cfl              = 0.35;
            obj.iterMax          = 10000;
            obj.timeMax          = 5;
            obj.bcMode.x         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            obj.bcInfinity       = 20;
            obj.activeSlices.xy  = false;
            obj.activeSlices.xyz = true;
%            obj.ppSave.dim2      = 10;
            obj.ppSave.dim3      = 20;
            obj.image.mass       = true;
            obj.image.interval   = 10;

%            obj.dGrid.x.points   = [0, 5;    33.3, 1;    66.6, 1;    100, 5];
            
            obj.logProperties    = [obj.logProperties, 'gamma'];

            obj.operateOnInput(input, [300, 6, 6]);
        end

    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
        % Returns the initial conditions for a corrugation shock wave according to the settings for
        % the initializer.
        % USAGE: [mass, mom, ener, mag, statics, run] = getInitialConditions();
        potentialField = [];
        selfGravity = [];        
        geo = obj.geomgr;

        geo.makeBoxSize([3 1.5 1.5]);

        P0 = .1;

        %--- Create and populate data arrays ---%
        [mass, mom, mag, ener] = geo.basicFluidXYZ();

        ener = ener * P0 / (obj.gamma-1);
        
        [x0 y0 z0] = obj.ndgridSetIJK('pos');

        cosa = 1; sina = 0;
        cosb = sqrt(1/5); sinb = sqrt(4/5);

        xpar = (cosb * x0 + sinb * y0)/(sqrt(4/5)*obj.grid(2)); % The only axis the CPAW depends on

        v1_0 = 0; % travelling CPAW
        b1_0 = 1;
        aamp = .1;

        mom(1,:,:,:) = mass.*(v1_0*cosb  + aamp*sin(2*pi*xpar)*sinb);
        mom(2,:,:,:) = mass.*(-v1_0*sinb + aamp*sin(2*pi*xpar)*cosb);
        mom(3,:,:,:) = mass.*(                                      aamp*cos(2*pi*xpar));

        mag(1,:,:,:) = b1_0*cosb  + aamp*sin(2*pi*xpar)*sinb;
        mag(2,:,:,:) = -b1_0*sinb + aamp*sin(2*pi*xpar)*cosb;
        mag(3,:,:,:) =                                       aamp*cos(2*pi*xpar);

        % Calculate the kinetic + magnetic contributions to energy density pre and postshock
        TM = squeeze(.5*sum(mom.^2,1))./mass + .5*squeeze(sum(mag.^2,1));
        ener = ener + TM;

        statics = StaticsInitializer();
        end


    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
