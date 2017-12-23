classdef ShearingBoxInitializer < Initializer
    
        
%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        innerRadius, outerRadius; % r- and r+ of the shearing annulus
        
        q; % rotation curve, w ~ r^-q
        Mstar; % mass and radius of the star
        Rstar;
        
        Sigma; % surface mass density at r=(inner radius + outer radius)/2
               % actually initializes if 2D, 
               % vertically-integrated with isothermal EOS is 3D
               
        dustFraction;
        
        densityExponent; % Sigma(r) = [r/(r- + r+)]^densityExponent: canonically 1.5
        
        rotationExponent; % omega ~ r^-q: canonically 1.5
        
        useZMirror;
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%____________________________________________________________________________ ShearingBoxInitializer
        function self = ShearingBoxInitializer(input)
            self                  = self@Initializer();            
            self.runCode          = 'SHEARBOX';
            self.mode.fluid       = true;
            self.mode.magnet      = false;
            self.mode.gravity     = true;
            self.iterMax          = 300;
            self.bcMode.x         = ENUM.BCMODE_CONSTANT; % FIXME wrong use outflow...
            self.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            self.bcMode.z         = ENUM.BCMODE_OUTFLOW;
            
            self.pureHydro = 1;
            
            self.numFluids = 2;
            
            self.Sigma = 1;
            self.dustFraction = .01;
            self.densityExponent = -1.5;
            self.rotationExponent = -0.5;
            
            self.gravity.constant = 1;
            self.Mstar            = 1;
            self.Rstar            = 0.3;
            self.gamma            = 7/5;
            
            self.useZMirror       = 0;

            self.operateOnInput(input, [64 64 1]);
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
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(self)

            geo = self.geomgr;
            nz = geo.globalDomainRez(3);
            
            %           self.bcMode.x = ENUM.BCMODE_CONSTANT;
            if self.useZMirror == 1
                if nz > 1
                    self.bcMode.z    = { ENUM.BCMODE_MIRROR, ENUM.BCMODE_OUTFLOW };
                else
                    if mpi_amirank0(); warning('NOTICE: .useZMirror was set, but nz = 1; Ignoring.\n'); end
                    self.bcMode.z = ENUM.BCMODE_CIRCULAR;
                end
            else
                if nz > 1
                    self.bcMode.z = ENUM.BCMODE_OUTFLOW;
                else
                    self.bcMode.z = ENUM.BCMODE_CIRCULAR;
                end
            end
            
            switch geo.pGeometryType;
                case ENUM.GEOMETRY_SQUARE;
                    boxsize = 2*(1+self.edgePadding)*diskInfo.rout * [1 1 1];
                    
                    if nz > 1
                        needheight = diskInfo.height * (1+self.edgePadding) * (2 - self.useZMirror);
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
                    
                    error('Shearing box does not support square coordinates at this time.');
                case ENUM.GEOMETRY_CYLINDRICAL;
                    width = self.outerRadius - self.innerRadius;
                    
                    dr = (width) / geo.globalDomainRez(1);
                    
                    if nz > 1
                        error('nz>1 broken with shearing bxo for now.');
                        
                        nz = geo.globalDomainRez(3);
                        availz = dr * nz;
                        needz = round(.5*(2 - self.useZMirror) * diskInfo.rout * (1+self.edgePadding) * diskInfo.aspRatio);
                        
                        if availz < needz
                            dz = dr * needz / availz;
                            if mpi_amirank0(); warning('NOTE: nz of %i insufficient to have dr = dz; Need %i; dz increased from r=%f to %f.', int32(nz), int32(ceil(geo.globalDomainRez(1)*needz/availz)), dr, dz); end
                        else
                            dz = dr;
                        end
                        
                        % For vertical mirror, offset Z=0 by three cells to agree with mirror BC that has 3 ghost cells
                        if self.useZMirror; z0 = -3*dz; else z0 = -round(nz/2)*dz; end
                    else
                        z0 = 0;
                        dz = 1;
                    end
                    
                    geo.geometryCylindrical(self.innerRadius, 1, dr, z0, dz)
            end
            
            [radpts, phipts, zpts] = geo.ndgridSetIJK('pos','cyl');
            
            r_c = (self.innerRadius + self.outerRadius)/2;
            
            % Pick G so that orbital period is unity
            %self.gravity.constant = 4*pi^2*r_c^3 / self.Mstar;\
            self.gravity.constant = 6.673e-11;
            cs_0 = 800;
            
            GM = self.gravity.constant * self.Mstar;
            
            if nz == 1    
                mass = self.Sigma * (radpts./r_c).^self.densityExponent;
                vel     = geo.zerosXYZ(geo.VECTOR);
                vel(2,:,:,:) = sqrt(GM) ./ sqrt(radpts);
                Eint = cs_0^2 * mass / .56; % FIXME HACK
                
                fluids(1) = self.rhoVelEintToFluid(mass, vel, Eint);
                self.fluidDetails(1) = fluidDetailModel('warm_molecular_hydrogen');
                
                mass = mass * self.dustFraction;
                Eint = .0001*geo.onesXYZ(geo.SCALAR);
                fluids(2) = self.rhoVelEintToFluid(mass, vel, Eint);
                self.fluidDetails(2) = fluidDetailModel('10um_iron_balls');
            else
                error('3d not supported yet lolz');
            end
            
            omegaInner = sqrt(GM / self.innerRadius^3);
            omegaOuter = sqrt(GM / self.outerRadius^3);
            self.frameParameters.rotateCenter = [0 0 0];
            self.frameParameters.omega = .5*(omegaOuter + omegaInner);
            
            % run for 100 years (@ this orbital radius)
            self.timeMax = 100*2*pi/self.frameParameters.omega;

            mag     = geo.zerosXYZ(geo.VECTOR);
                          
            statics = [];%StaticsInitializer(self.grid);

            selfGravity = [];
            %starX = (self.grid+1)*dGrid/2;
            %selfGravity = SelfGravityInitializer();
            %selfGravity.compactselfectStates = [1 self.pointRadius starX(1) starX(2) starX(3) 0 0 0 0 0 0];
                                           % [m R x y z vx vy vz lx ly lz]
            
            potentialField = PotentialFieldInitializer();

            sphericalR = sqrt(radpts.^2 + zpts.^2);
            potentialField.field = -self.gravity.constant*self.Mstar./sphericalR;
            
            % Constructs a single-parameter softened potential -a/sqrt(r^2 + r0^2) inside pointRadius to avoid
            % a singularity approaching r=0; 'a' is chosen sqrt(2) to match external 1/r
            soft = (sphericalR < self.Rstar);
            phiSoft = -sqrt(2./(sphericalR(soft).^2 + self.Rstar^2));
            
            potentialField.field(soft) = phiSoft;
            
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
