classdef RayleighTaylorInitializer < Initializer
% Run a simulation of the RT instability to test Imogen
%
%   useStatics        specifies if static conditions should be set for the run.       logical  
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    rhoTop;
    rhoBottom;
    P0;
    Bx;
    Bz;

    Kx;
    Ky;
    Kz;
    pertAmplitude;
    randomPert;

    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]

    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]

    end %PROTECTED

%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ GravityTestInitializer
        function obj = RayleighTaylorInitializer(input)            
            obj                     = obj@Initializer();
            obj.grid = input;
            obj.runCode             = 'RAYLEIGH_TAYLOR';
            obj.info                = 'Rayleigh-Taylor instability test';
            obj.mode.fluid          = true;
            obj.mode.magnet         = false;
            obj.mode.gravity        = true;
            obj.iterMax             = 300;
            obj.gamma               = 1.4;
            obj.bcMode.x            = 'circ';
            obj.bcMode.y            = 'fade';
            obj.bcMode.z            = 'circ';

            obj.bcInfinity          = 4;

            obj.activeSlices.xy     = true;
            obj.timeUpdateMode      = ENUM.TIMEUPDATE_PER_STEP;
            
            obj.gravity.constant    = 1;
            obj.gravity.solver      = ENUM.GRAV_SOLVER_EMPTY;

            obj.rhoTop              = 2;
            obj.rhoBottom           = 1;
            obj.P0                  = 2.5;
            obj.Bx                  = 0;
            obj.Bz		    = 0;

            obj.pertAmplitude = .0001;
            obj.Kx = 1;
            obj.Ky = 1;
            obj.Kz = 1;
            obj.randomPert = 0;
            
            obj.operateOnInput(input, [200, 100, 100]);
        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [mass, mom, ener, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            %--- Ensure that the grid dimensions are even. ---%
            %       Even grid size means that the star will be correctly placed in the center cell.
            GIS = GlobalIndexSemantics();
            potentialField = [];
            selfGravity = [];
            %obj.grid -> GIS.pMySize; this enables parallelization
            
            [X Y Z] = GIS.ndgridSetXYZ();
            obj.bcMode.grav.y = ENUM.BCMODE_FADE;
            
            % Establish low density below, high density above
            %mass = zeros(GIS.pMySize);
            %mass(:,1:GIS.pMySize(2)/2,:)       = obj.rhoBottom;
            %mass(:,(GIS.pMySize(2)/2+1):end,:) = obj.rhoTop;

            % Compute pressure & gravity field to establish hydrostatic equilibrium
            %[Xg Yg Zg] = ndgrid(1:GIS.pMySize(1), 1:GIS.pMySize(2), 1:GIS.pMySize(3));
            %obj.dGrid   = [.5 (.5*GIS.pMySize(2)/GIS.pMySize(1)) .5] ./ GIS.pMySize;
            %Xg = Xg * obj.dGrid(1);
            %Yg = Yg * obj.dGrid(2);
            %Zg = Zg * obj.dGrid(3);
            %Y0 = Yg(1,round(end/2),1);
            
            obj.dGrid   = [.5 (.5*obj.grid(2)/obj.grid(1)) .5] ./ obj.grid;
            X = X * obj.dGrid(1);
            Y = Y * obj.dGrid(2);
            Z = Z * obj.dGrid(3);
            Y0 = .5*obj.dGrid(2)*obj.grid(2);
            
            % Establish low density below, high density above
            mass             = zeros(GIS.pMySize);
            mass(Y < Y0)     = obj.rhoBottom;
            mass(Y >= Y0)    = obj.rhoTop;

            % Gas pressure to balance gravity
            ener = (obj.P0 - obj.gravity.constant * (cumsum(mass,2)-mass(1,1,1)) * obj.dGrid(2) ) / (obj.gamma-1);
            %copmute energy density locally in each cell:
            %ener = (pressure@cell - obj.gravity.constant * (cumsum(mass,2)-mass(1,1,1)) * obj.dGrid(2) ) / (obj.gamma-1)

            obj.gravity.fixedPotential = obj.gravity.constant * Y;

            mom         = zeros([3 GIS.pMySize]);
            obj.thresholdMass = .0001 * obj.rhoBottom;
            obj.minMass = .5*obj.thresholdMass;
            mag         = zeros([3 GIS.pMySize]);

            if obj.randomPert == 0
                % Set initial velocity perturbation
                % Vy = A [1+cos(2pi Kx x/Lx)] [1+cos(2pi Ky y/Ly)]/4
                if GIS.pMySize(3) == 1;
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5))/ 4;
                else
                    mom(2,:,:,:) = obj.pertAmplitude * (1+cos(2*pi*obj.Kx*X/.5)) .* (1+cos(2*pi*obj.Ky*(Y-Y0)/1.5)) .* (1+cos(2*pi*obj.Kz*Z/.5))/ 8;
                end
            else
                w = (rand([GIS.pMySize(1) GIS.pMySize(3)])-.5) * obj.pertAmplitude;
                for y = 1:GIS.pMySize(2); mom(2,:,y,:) = w; end
            end

            mom(2,:,:,:) = squeeze(mom(2,:,:,:)).*mass;

            % Kinetic energy density
            ener = ener + .5 * squeeze(mom(2,:,:,:).^2) ./ mass;

            % If doing magnetic R-T, turn on magnetic flux & set magnetic field & add magnetic energy
            if (obj.Bx ~= 0.0) || (obj.Bz ~= 0.0)
                obj.mode.magnet = true;
                mag(1,:,:,:) = obj.Bx;
		mag(3,:,:,:) = obj.Bz;

                ener = ener + .5 * squeeze(sum(mag.^2, 1));
            end

%statics = [];            return;
            
            yfix = [(GIS.pMySize(2)-3):GIS.pMySize(2)];
            
            %%%% Force top and bottom edges to be static
            statics.values  = [0, obj.rhoBottom, obj.rhoTop, ener(1,3,1), ener(1,end-3,1), obj.Bx, obj.Bz];
            statics.mass.s  = uint8(zeros(GIS.pMySize));
            statics.mass.s(:,1:3,:) = 2;
            statics.mass.s(:,yfix,:) = 3;

            statics.ener.s  = uint8(zeros(GIS.pMySize));
            statics.ener.s(:,1:3,:) = 4;
            statics.ener.s(:,yfix,:) = 5;

            fields = {'x', 'y', 'z'};
            for i=1:3
                statics.mom.s.(fields{i}) = uint8(zeros(GIS.pMySize));
		statics.mom.s.(fields{i})(:,1:3,:) = 1;
                statics.mom.s.(fields{i})(:,yfix,:) = 1;
            end

	    i = 1;
            statics.mag.s.(fields{i}) = uint8(zeros(GIS.pMySize));
            statics.mag.s.(fields{i})(:,1:3,:) = 6;
            statics.mag.s.(fields{i})(:,yfix,:) = 6;

            i = 3;
            statics.mag.s.(fields{i}) = uint8(zeros(GIS.pMySize));
            statics.mag.s.(fields{i})(:,1:3,:) = 7;
            statics.mag.s.(fields{i})(:,yfix,:) = 7;

        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
