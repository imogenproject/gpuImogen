classdef JetInitializer < Initializer
% Creates imogen input data arrays for the fluid, magnetic-fluid, and gravity jet tests.
%
% Unique properties for this initializer:
%   direction  % enumerated direction of the jet.                                  str
%   jetMass    % mass value of the jet.                                            double
%   jetMach    % mach speed for the jet.                                           double
%   jetMags    % magnetic field amplitudes for the jet.                            double(3)
%   injectorOffset     % index location of the jet on the grid.                            double(3)
%   backMass   % mass value in background fluid.                                   double
%   backMags   % magnetic field amplitudes in background fluid.                    double(3)
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        X = 'x';
        Y = 'y';
        Z = 'z';
        
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        direction;  % enumerated direction of the jet.                                  str
        flip;       % specifies if the jet should be negatively directed.               logical
        injectorSize;

        jetMass;    % mass value of the jet.                                            double
        jetMach;    % mach speed for the jet.                                           double
        jetMags;    % magnetic field amplitudes for the jet.                            double(3)
        injectorOffset;     % index location of the jet on the grid.                            double(3)
        backMass;   % mass value in background fluid.                                   double
        backMags;   % magnetic field amplitudes in background fluid.                    double(3)
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
        function obj = JetInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'Jet';
            obj.info             = 'Jet trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.cfl              = 0.85;
            obj.iterMax          = 250;
            obj.bcMode.x         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.y         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            obj.activeSlices.xy  = true;
            obj.ppSave.dim2      = 10;
            
            obj.direction        = JetInitializer.X;
            obj.flip             = false;
            obj.injectorSize     = 2;

            obj.jetMass          = 1;
            obj.jetMach          = 1;
            obj.jetMags          = [0 0 0];
            obj.backMass         = 1;
            obj.backMags         = [0 0 0];
            
            obj.operateOnInput(input, [512 256 1]);
        end
        
%___________________________________________________________________________________________________ jetMags
        function result = get.jetMags(obj)
            result = obj.make3D(obj.jetMags, 0);
        end
           
%___________________________________________________________________________________________________ jetMags
        function result = get.backMags(obj)
            result = obj.make3D(obj.backMags, 0);
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
            
            geo = obj.geomgr;
            rez = geo.globalDomainRez;
            
            if isempty(obj.injectorOffset)
                obj.injectorOffset = [ceil(rez(1)/10), ceil(rez(2)/2), ceil(rez(3)/2)];
            end
               
            % Box height = 1, width = aspect ratio
            obj.geomgr.makeBoxSize(rez(1)/rez(2));

            [mass, mom, mag, ener] = obj.geomgr.basicFluidXYZ();
            
            %--- Magnetic background ---%
            for i=1:3;    mag(i,:,:,:) = obj.backMags(i)*ones(obj.geomgr.localDomainRez); end
            
            %--- Total energy ---%
            magSquared    = squish( sum(mag .* mag, 1) );
            ener          = (mass.^obj.gamma)/(obj.gamma - 1) + 0.5*magSquared;
            
            %--- Static values for the jet ---%
            jetMom            = obj.jetMass*speedFromMach(obj.jetMach, obj.gamma, obj.backMass, ...
                                                            ener(1,1,1), obj.backMags');

            if (obj.flip), jetMom = - jetMom; end

            jetEner = 1*(obj.backMass^obj.gamma)/(obj.gamma - 1) ...   % internal
                        + 0.5*(jetMom^2)/obj.jetMass ...            % kinetic
                        + 0.5*sum(obj.jetMags .* obj.jetMags, 2);   % magnetic

            statics = StaticsInitializer(geo);
             
            xMin = max(obj.injectorOffset(1)-2,1);        xMax = min(obj.injectorOffset(1)+2,rez(1));
            yMin = max(obj.injectorOffset(2)-obj.injectorSize(2),1);        yMax = min(obj.injectorOffset(2)+obj.injectorSize(2),rez(2));
            zMin = max(obj.injectorOffset(3)-obj.injectorSize(2),1);        zMax = min(obj.injectorOffset(3)+obj.injectorSize(2),rez(3));

            statics.valueSet = {0, obj.jetMass, jetMom, jetEner, obj.jetMags(1), ...
                            obj.jetMags(2), obj.jetMags(3), obj.backMass, ener(1,1)};

            %iBack = statics.indexSetForCube( (xMin-2):(xMin-1),(yMin-2):(yMax+2), zMin:zMax);
            %iTop  = statics.indexSetForCube( xMin:xMax,    (yMax+1):(yMax+2), zMin:zMax);
            %iBot  = statics.indexSetForCube( xMin:xMax,    (yMin-2):(yMin-1), zMin:zMax);

            %injCase = [iBack; iTop; iBot];
            
            caseFcn = @(x, y, z) obj.cellIsInjectorCase(obj.injectorOffset, obj.injectorSize, x, y, z);
            coreFcn = @(x, y, z) obj.cellIsInjectorCore(obj.injectorOffset, obj.injectorSize, x, y, z);

            x0 = obj.injectorOffset; r = obj.injectorSize(1); d = obj.injectorSize(2);
            
            
            injCase = statics.indexSetForFunction(caseFcn, [x0(1) - d, x0(1)+d], [x0(2) - r-d, x0(2)+r+d], [x0(3)-r-d, x0(3)+r+d]);
            injCore = statics.indexSetForFunction(coreFcn, [x0(1) - d, x0(1)+d], [x0(2) - r-d, x0(2)+r+d], [x0(3)-r-d, x0(3)+r+d]);
            
            statics.indexSet = { injCore, injCase };
            
            %statics.indexSet = {statics.indexSetForCube(xMin:xMax,yMin:yMax,zMin:zMax), injCase, statics.indexSetForCube( (rez(1)-1):rez(1),1:rez(2),1) };
            %statics.indexSet{4} = statics.indexSetForCube(1:(rez(1)-2), 1:2, 1);

            statics.associateStatics(ENUM.MASS, ENUM.SCALAR,   statics.CELLVAR, 1, 2);
            statics.associateStatics(ENUM.ENER, ENUM.SCALAR,   statics.CELLVAR, 1, 4);
            statics.associateStatics(ENUM.MOM, ENUM.VECTOR(1), statics.CELLVAR, 1, 3);

            statics.associateStatics(ENUM.MASS, ENUM.SCALAR,   statics.CELLVAR, 2, 8);
            statics.associateStatics(ENUM.MOM, ENUM.VECTOR(1), statics.CELLVAR, 2, 1);
            statics.associateStatics(ENUM.MOM, ENUM.VECTOR(2), statics.CELLVAR, 2, 1);

            if obj.mode.magnet
                statics.associateStatics(ENUM.MAG, ENUM.VECTOR(1), statics.CELLVAR, 1, 5);
                statics.associateStatics(ENUM.MAG, ENUM.VECTOR(2), statics.CELLVAR, 1, 6);
                statics.associateStatics(ENUM.MAG, ENUM.VECTOR(3), statics.CELLVAR, 1, 7);
            end

            if obj.mode.magnet;     obj.runCode = [obj.runCode 'Mag'];  end
            if obj.mode.gravity;    obj.runCode = [obj.runCode 'Grav']; end

            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
        end
        
%___________________________________________________________________________________________________ toInfo
        function result = toInfo(obj)
            skips = {'X', 'Y', 'Z'};
            result = toInfo@Initializer(obj, skips);
        end                    
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
        
        function yesno = cellIsInjectorCase(offset, dims, x, y, z)
            % given an offset [x0 y0 z0] and dimensions [rin, wall_thickness],
            % returns ONE iff,
            % sqrt((y-y0)^2+(z-z0)^2) > dims(1), < dims(2) AND ( (x >= x0) OR (x < (x0+depth))
            % (the circular cowling)
            % OR
            % sqrt((y-y0)^2+(z-z0)^2) < dims(2) and (x >= (x0-depth)) OR (x < x0) )
            % (the circular bottom plate)
            yesno = 0;
            
            r = sqrt((y-offset(2))^2+(z-offset(3))^2);
            rout = dims(1)+dims(2);
            x = x - offset(1);
            if (r > dims(1)) && (r < rout) && ( (x >= 0) || (x < (0+dims(2))) ); yesno = 1; end
            
            if (r < rout) && (x < 0) && (x > (0 - dims(2))); yesno = 1; end
            
        end
        
        function yesno = cellIsInjectorCore(offset, dims, x, y, z)
            % given an offset [x0 y0 z0] and dimensions [rin, rout, depth],
            % returns ONE iff,
            % sqrt((y-y0)^2+(z-z0)^2) <= dims(1) AND ( (x >= x0) OR (x < (x0+depth))
            % (the circular cowling)
            % OR
            % sqrt((y-y0)^2+(z-z0)^2) < dims(2) and (x >= (x0-depth)) OR (x < x0) )
            % (the circular bottom plate)
            yesno = 0;
            
            r = sqrt((y-offset(2))^2+(z-offset(3))^2);
            if (r <= dims(1))  && ( (x >= offset(1)) || (x < (offset(1)-dims(2))) ); yesno = 1; end
        end
    end
end%CLASS
