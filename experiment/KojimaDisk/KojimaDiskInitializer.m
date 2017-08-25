classdef KojimaDiskInitializer < Initializer
% Uses equilibrium routines to create an equilibrium disk configuration and then formats them for
% use in Imogen based on the Kojima model. The disk is created in the polytropic unit systems under
% the constraints that the gravitational constant, G, the mass of the central star, M, and the 
% polytropic constant, K, are all unity.
%
% Unique properties for this initializer:
%   q                 angular velocity exponent (omega = omega_0 * r^-q.              double
%   radiusRatio       (inner disk radius) / (radius of density max).                  double
%   edgePadding       number of cells around X & Y edges to leave blank.              double
%   pointRadius       size of the softened point at the center of the grid            double
%   useStatics        specifies if static conditions should be set for the run.       logical
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
        MOMENTUM_DISTRIB_NONE   = 1;
        MOMENTUM_DISTRIB_KOJIMA = 2;
        MOMENTUM_DISTRIB_KEPLER = 3;
        MOMENTUM_DISTRIB_SOFTEN = 4;
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        q;              % angular velocity exponent; omega = omega_0 * r^-q.            double
        radiusRatio;    % inner radius / radius of density max.                         double
        edgePadding;    % number of cells around X & Y edges to leave blank.            double
        pointRadius;    % size of the softened point at the center of the grid.         double
        diskMomDist;    % 1x4 describes momentum distribution                double [4]
        pointMass;

        bgDensityCoeff; % Min density is this times max initial density            double
        useZMirror;     % If 1 and run is 3d, simulates the top half of the disk only   logical

        useStatics;     % specifies if static conditions should be set for the run.     logical
        inflatePressure;% artificially increase the background pressure.                logical 

        perturbDisk;    % If true, the disk generated is disturbed away from equilibrium
        perturbRhoAmp; 

        dustLoad; 

        buildShearingAnnulus;
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ KojimaDiskInitializer
        function obj = KojimaDiskInitializer(input)
            obj                  = obj@Initializer();            
            obj.runCode          = 'KOJIMA';
            obj.info             = 'Kojima point-potential disk trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = true;
            obj.iterMax          = 300;
            obj.bcMode.x         = ENUM.BCMODE_CONSTANT; % FIXME wrong use outflow...
            obj.bcMode.y         = ENUM.BCMODE_CONSTANT;
            obj.activeSlices.xy  = true;
            obj.bgDensityCoeff   = 1e-5;
            
            obj.gravity.constant = 1;
            obj.pointMass        = 1;
            obj.pointRadius      = 0.3;
            obj.gamma            = 5/3;
            obj.q                = 2;
            obj.radiusRatio      = 0.8;
            obj.edgePadding      = 0.5;

            obj.thresholdMass    = 0;
            obj.useStatics       = false;
            obj.inflatePressure  = false;
            obj.useZMirror       = 0;

            obj.perturbDisk      = 0;
            obj.perturbRhoAmp    = 0;
            
            obj.buildShearingAnnulus = 0;
            obj.dustLoad         = 0;
            %--- Set momentum distribution array ---%
            %           This array defines how Imogen distributes momentum in the grid. A value of 1 
            %           specifies no momentum, 2 gives Kojima r^1-q momentum, 3 gives Keplerian 
            %           r^-.5, 4 gives a momentum appropriate to the softened potential at the 
            %           center.
            %
            %           First number selects momentum for 0-pointradius.
            %           Second selects momentum for pointradius - radiusRatio.
            %           Third selects momentum for the disk itself.
            %           Fourth selects momentum outside the disk.
            %           Default is zero momentum except in the disk.
            %           In principle [4 3 2 3] is nearest equilibrium but it is violently unstable.
            obj.diskMomDist         = [ KojimaDiskInitializer.MOMENTUM_DISTRIB_NONE, ...
                                        KojimaDiskInitializer.MOMENTUM_DISTRIB_NONE, ...   
                                        KojimaDiskInitializer.MOMENTUM_DISTRIB_KOJIMA, ...
                                        KojimaDiskInitializer.MOMENTUM_DISTRIB_NONE ];
            
            obj.operateOnInput(input, [64 64 1]);
            
        end
        
%___________________________________________________________________________________________________ GS: pointMass
% Dynmaic pointMass property to connect between the gravity vars structure and run files.
    
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]                
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)

            geo = obj.geomgr;
            nz = geo.globalDomainRez(3);
            
            %           obj.bcMode.x = ENUM.BCMODE_CONSTANT;
            if obj.useZMirror == 1
                if nz > 1
                    obj.bcMode.z    = { ENUM.BCMODE_MIRROR, ENUM.BCMODE_OUTFLOW };
                else
                    if mpi_amirank0(); warning('NOTICE: .useZMirror was set, but nz = 1; Ignoring.\n'); end
                end
            end
            
            diskInfo = kojimaDiskParams(obj.q, obj.radiusRatio, obj.gamma);

            switch geo.pGeometryType;
                case ENUM.GEOMETRY_SQUARE;
                    boxsize = 2*(1+obj.edgePadding)*diskInfo.rout * [1 1 1];
                    
                    if nz > 1
                        needheight = diskInfo.height * (1+obj.edgePadding) * (2 - obj.useZMirror);
                        gotheight  = boxsize * geo.globalDomainRez(3) / geo.globalDomainRez(1);
                        
                        if needheight > gotheight
                            warning('NOTE: dz = dx will not be tall enough to hold disk; Raising dz.');
                            boxsize(3) = needheight;
                        else
                            boxsize(3) = gotheight;
                        end
                        
                    end
                    
                    geo.makeBoxSize(boxsize);
                    geo.makeBoxOriginCoord(round(geo.globalDomainRez/2)+0.5);
                case ENUM.GEOMETRY_CYLINDRICAL;
                    width = diskInfo.rout - diskInfo.rin;
                    inside = diskInfo.rin - obj.edgePadding * width;
                    outside = diskInfo.rout + obj.edgePadding * width;
                    
                    dr = (outside - inside) / geo.globalDomainRez(1);
                    
                    if nz > 1
                        nz = geo.globalDomainRez(3);
                        availz = dr * nz;
                        needz = round(.5*(2 - obj.useZMirror) * diskInfo.rout * (1+obj.edgePadding) * diskInfo.aspRatio);
                        
                        if availz < needz
                            dz = dr * needz / availz;
                            
                            if mpi_amirank0(); warning('NOTE: nz of %i insufficient to have dr = dz; Need %i; dz increased from r=%f to %f.', int32(nz), int32(ceil(geo.globalDomainRez(1)*needz/availz)), dr, dz); end
                        else
                            dz = dr;
                        end
                        
                        % For vertical mirror, offset Z=0 by three cells to agree with mirror BC that has 3 ghost cells
                        if obj.useZMirror; z0 = -3*dz; else z0 = -round(nz/2)*dz; end
                    else
                        z0 = 0;
                        dz = 1;
                    end
                    
                    geo.geometryCylindrical(inside, 1, dr, z0, dz)
            end

            mom     = geo.zerosXYZ(geo.VECTOR);

            [radpts, phipts, zpts] = geo.ndgridSetIJK('pos','cyl');
            [mass, momA, momB, Eint] = evaluateKojimaDisk(obj.q, obj.gamma, obj.radiusRatio, 1, obj.bgDensityCoeff, radpts, phipts, zpts, geo.pGeometryType);

            obj.minMass = mpi_max(max(mass(:))) * obj.bgDensityCoeff;
            mass    = max(mass, obj.minMass);

            if obj.perturbDisk
                m0 = obj.perturbRhoAmp * (rand(size(mass)) - 0.5);
                fullydisk = (mass > 100*obj.minMass);
                mass(fullydisk) = mass(fullydisk) .* (1 + m0(fullydisk));
            end

            mag     = geo.zerosXYZ(geo.VECTOR);
            
            if obj.inflatePressure
                minDiskMass = minFinderND(mass(mass > obj.minMass));
            else
                minDiskMass = obj.minMass;
            end
            
            ener    = Eint ...   % internal energy
                        + 0.5*(momA.^2+momB.^2) ./ mass;% ...           % kinetic energy
                        %+ 0.5*squish(sum(mag .* mag, 1));                      % magnetic energy                    
            
            mom(1,:,:,:) = momA;
            mom(2,:,:,:) = momB;
            statics = [];%StaticsInitializer(obj.grid);

            selfGravity = [];
            %starX = (obj.grid+1)*dGrid/2;
            %selfGravity = SelfGravityInitializer();
            %selfGravity.compactObjectStates = [1 obj.pointRadius starX(1) starX(2) starX(3) 0 0 0 0 0 0];
                                           % [m R x y z vx vy vz lx ly lz]

            potentialField = PotentialFieldInitializer();

            fluids = obj.stateToFluid(mass, mom, ener);

            if obj.dustLoad > 0
                
            end

            sphericalR = sqrt(radpts.^2 + zpts.^2);
            potentialField.field = -1./sphericalR;
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
