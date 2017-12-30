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
        
        Sigma; % surface gas mass density at r=(inner radius + outer radius)/2
        dustFraction; % scales initial rho_dust = dustFraction x rho_gas
        
        densityExponent; % Sigma(r) = [r/(r- + r+)]^densityExponent: canonically 1.5
        temperatureExponent; % omega ~ r^-q: canonically 1.5 (NSG, no radial pressure gradient)
        
	densityCutoffFraction;

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
            self.bcMode.z         = ENUM.BCMODE_CONSTANT;
            
            self.pureHydro = 1;
            
            self.numFluids = 2;
            
            self.Sigma = 1;
            self.dustFraction = .01;
            self.densityExponent = -1.5;
            self.temperatureExponent = -0.5;

	    self.densityCutoffFraction = 1e-6;
            
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
                    self.bcMode.z = ENUM.BCMODE_CONSTANT;
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
                        nz = geo.globalDomainRez(3);
                        %availz = dr * nz;
                        
                        %if availz < needz
                        %    dz = dr * needz / availz;
                        %    if mpi_amirank0(); warning('NOTE: nz of %i insufficient to have dr = dz; Need %i; dz increased from r=%f to %f.', int32(nz), int32(ceil(geo.globalDomainRez(1)*needz/availz)), dr, dz); end
                        %else
                            dz = dr;
                        %end
                        
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
            
            radpts = radpts / r_c; % normalize radius
	    zpts = zpts / r_c;     % normalize height

            self.gravity.constant = 6.673e-11;
            cs_0 = 800; % isothermal c_s picked rather arbitrarily...
            
            GM = self.gravity.constant * self.Mstar;
            
            % Calculate the isothermal thin-disk scale height H = c_isothermal / omega_kepler;
	    w0 = sqrt(GM/r_c^3);
	    h0 = cs_0 / w0;

            scaleHeight = cs_0 *radpts.^(1.5+.5*self.temperatureExponent) / sqrt(GM);
            
            % calculate rho at midplane
	    q = (-self.temperatureExponent - 3 + 2*self.densityExponent)/2;
	    rho_0 = self.Sigma / (sqrt(2*pi)*h0);
	    rho_mp = rho_0 * radpts.^q;

	    phi_0 = -GM / r_c;

	    deltaphi = -phi_0*(1/sqrt(radpts.^2 + zpts.^2) - 1./radpts);
	    mass = rho_mp .* exp(deltaphi .* radpts.^self.temperatureExponent / cs_0^2);

            % asymptotic result for very thin disk (H/R << 1) = vertical gaussian
            %mass = self.Sigma * (radpts./r_c).^self.densityExponent .* exp(-(zpts./scaleHeight).^2/2) ./(scaleHeight * sqrt(2*pi));

            self.fluidDetails(1) = fluidDetailModel('warm_molecular_hydrogen'); 
            self.fluidDetails(1).minMass = mpi_max(max(mass(:))) * self.densityCutoffFraction;
            vel     = geo.zerosXYZ(geo.VECTOR);

            vel(2,:,:,:) = r_c*radpts.*sqrt((cs_0^2*q/r_c^2)*radpts.^(self.temperatureExponent-2) + w0^2*radpts.^-3); 
            velB = sqrt(GM) ./ sqrt(radpts);
            Eint = cs_0^2 * mass .* radpts.^self.temperatureExponent / (self.fluidDetails(1).gamma-1); % FIXME HACK
            
            fluids(1) = self.rhoVelEintToFluid(mass, vel, Eint);

            % assume uniform dispersal of dust through gas
            mass = mass * self.dustFraction;
            self.fluidDetails(2) = fluidDetailModel('10um_iron_balls');
            self.fluidDetails(2).minMass = mpi_max(max(mass(:))) * self.densityCutoffFraction;
            Eint = mass * (.01*cs_0)^2 / (self.fluidDetails(2).gamma - 1);
            fluids(2) = self.rhoVelEintToFluid(mass, vel, Eint);
            
            velInner = sqrt(GM / self.innerRadius);
            velOuter = sqrt(GM / self.outerRadius);
            self.frameParameters.rotateCenter = [0 0 0];
            self.frameParameters.omega = (velInner + velOuter)/(self.innerRadius + self.outerRadius);
            
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

            sphericalR = r_c*sqrt(radpts.^2 + zpts.^2);
            potentialField.field = -GM ./ sphericalR;
            
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
