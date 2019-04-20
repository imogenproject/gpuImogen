classdef ShearingBoxInitializer < Initializer
% The ShearingBoxInitializer simulates part of a (possibly dusty) protoplanetary disk.
% The gas density is derived from a surface density ansatz,
%     sigma(r) = Sigma0 * (r/r0)^p
% where p is canonically -1 (steady state thin alpha-disk)
% Temperature is set on cylinders following another radial power law,
%     T(r) = T0 * (r/r0)^n
% with n canonically -0.5 (from P_sunlight ~ r^-2 = P_blackbody ~ T^4)
% From this the scale height is derived as a function of radius
%     H(r) = (cs0 / omega0)*(r/r0)^(n/2 + 3/2)
% and the 3D density is written
%     rho(r,z) = rho0(r) exp(-alpha phi0(1/r - 1/R)/ cs0^2 r^n)
% where alpha = (1+dustFraction) and phi0 = GM/r0 and R = sqrt(r^2+z^2) is the spherical
% radius and rho0(r) = sigma(r) / (sqrt(2*pi) H(r)) is the midplane density. For a very thin
% disk, the choice of rho0 means that the vertical Gaussian integral returns sigma. The
% actual analytic result is only close to Gaussian.
% The rotation function is rotates-on-rings and v_phi(r,z) is derived from the mechanical
% balance equation,
%     F_radial = alpha rho v^2 / r - GM r^2/R^3 - (dP/dr)/r = 0
%
% If dust is added, alpha is greater than 1 which modifies the density/velocity functions to
% preserve equilibrium under the assumption of perfect coupling (Vgas-Vdust = 0). The actual
% time-independent equilibrium for the dust is a vertical delta function but this is not seen
% as a reasonable initial condition. With dust, small radial/vertical motions occur; If
% dustFraction is set to zero, the gas-only IC is seen to converge to a true time-independent
% solution upon grid refinement.
%
% It is important to note: the thermodynamic parameter inputs (geometric x-section and molecular
% mass) are in *SI UNITS*. A dusty disk must be specified in SI units if it is normalized
% because the normalization rescales molecular properties assuming they are SI.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]

    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        innerRadius, outerRadius; % r- and r+ of the annulus

        Mstar; % Stellar mass. Default = 1.981e30 (Msun)
        Rstar; % Stellar radius. Default = 1e9 (Rsun)

        normalizationRadius; % r0
        normalizeValues; % Rescales such that r0 = 1, rho(r0) = 1, GM/r0 = 1.
        
        Sigma0; % Gas surface mass density at r=r0
        densityExponent; % Sigma0(r) = Sigma0 * (r/r0)^q : canonically -1 (steady state accretion)
        densityCutoffFraction;

        dustFraction; % scales initial rho_dust = dustFraction x rho_gas

        dustPerturb; % set an initial magnitude delta v for the dust if > 0
        gasPerturb; % set an initial magnitude delta v for the gas if > 0
        
        cs0; % Isothermal soundspeed kb T0 / mu evaluated at r0
        temperatureExponent; % T(r) = T(r0) (r/r0)^n: canonically -.5 (purely starlight heating)
        

        useZMirror;
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        azimuthalMode;
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        pAzimuthalMode;
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]

        function set.azimuthalMode(self, m)
            if m >= 1
                self.pAzimuthalMode = m;
            else
                Save.logPrint('Error: cylindrical azimuthal symmetry mode must be >= 1, rx''d %f\n', m)
            end
        end

        function m = get.azimuthalMode(self)
            m = self.pAzimuthalMode;
        end

%____________________________________________________________________________ ShearingBoxInitializer
        function self = ShearingBoxInitializer(input)
            if nargin == 0; input = [64 64 1]; end
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
            
            self.innerRadius = 5*150e9; % 5 to 20 AU
            self.outerRadius = 20*150e9; 

            self.normalizationRadius = 150e10; % Normalize by values for 10AU
            self.normalizeValues = 1; 

            self.Sigma0 = 1000; % 100g/cm^2 = 1000kg/m^2
            self.densityExponent = -1.5;
            self.dustFraction = .01;

            self.dustPerturb = 0;
            self.gasPerturb  = 0;

            self.temperatureExponent = -0.5;
            self.cs0 = 1137; % isothermal soundspeed of 75% H_2 25% 4He at 273.15K

            self.densityCutoffFraction = 1e-5;
            
            self.gravity.constant = 1;
            self.Mstar            = 1.981e30;
            self.Rstar            = 1e9; 
            self.gamma            = 7/5;
            
            self.useZMirror       = 0;

            self.azimuthalMode = 1; % global

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

            r_c    = self.normalizationRadius;
           
            if self.useZMirror == 1
                if nz > 1
                    self.bcMode.z    = { ENUM.BCMODE_MIRROR, ENUM.BCMODE_FREEBALANCE };
                else
                    SaveManager.logPrint('    NOTICE: .useZMirror was set, but nz = 1; Ignoring.\n');
                    self.bcMode.z = ENUM.BCMODE_CIRCULAR;
                end
            else
                if nz > 1
                    %self.bcMode.z = ENUM.BCMODE_STATIC;
                    %self.bcMode.z = ENUM.BCMODE_OUTFLOW;
                    self.bcMode.z = ENUM.BCMODE_FREEBALANCE;
                else
                    self.bcMode.z = ENUM.BCMODE_CIRCULAR;
                end
            end
            
            switch geo.pGeometryType
                case ENUM.GEOMETRY_SQUARE
                    % fixme this is gonna dump if it's ever run again
                    boxsize = 2*(1+self.edgePadding)*diskInfo.rout * [1 1 1];
                    
                    if nz > 1
                        needheight = diskInfo.height * (1+self.edgePadding) * (2 - self.useZMirror);
                        gotheight  = boxsize * geo.globalDomainRez(3) / geo.globalDomainRez(1);
                        
                        if needheight > gotheight
                            warning('    NOTE: dz = dx will not be tall enough to hold disk; Raising dz.');
                            boxsize(3) = needheight;
                        else
                            boxsize(3) = gotheight;
                        end
                        
                    end
                    
                    geo.makeBoxSize(boxsize);
                    geo.makeBoxOriginCoord(round(geo.globalDomainRez/2)+0.5);
                    
                    error('Shearing box does not support square coordinates at this time.');
                case ENUM.GEOMETRY_CYLINDRICAL
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
                    
                        % FIXME this needs to autodetect the number of ghost cells in use
                        % For vertical mirror, offset Z=0 by four cells to agree with mirror BC that has 4 ghost cells
                        if self.useZMirror; z0 = -4*dz; else; z0 = -round(nz/2)*dz; end
                    else
                        z0 = 0;
                        dz = 1;
                    end
                    
                    geo.geometryCylindrical(self.innerRadius/r_c, self.azimuthalMode, dr/r_c, z0/r_c, dz/r_c);
                    SaveManager.logPrint('    Using cylindrical geometry; Angular slice set to %fdeg\n', 360/self.azimuthalMode);
%                    geo.geometryCylindrical(self.innerRadius, 1, dr, z0, dz)
            end

            % Fetch and normalize coordinates
            [radpts, phipts, zpts] = geo.ndgridSetIJK('pos','cyl');
            rsph   = sqrt(radpts.^2+zpts.^2);
clear phipts; % this is never used
clear zpts; % never used again

            self.gravity.constant = 6.673e-11; % physical value
            GM = self.gravity.constant * self.Mstar;
            
            cs_0 = self.cs0; % isothermal c_s picked rather arbitrarily...

            % Calculate several dimensionful scale factors
            w0    = sqrt(GM/r_c^3); % kepler omega @ r_c
            v0    = r_c*w0;         % Kepler orbital velocity @ r_c
            h0    = cs_0 / w0;      % scale height @ r_c
            phi_0 = -GM / r_c;      % stellar potential @ r_c
            rho_0 = self.Sigma0 / (sqrt(2*pi)*h0); % characteristic density @ r_c
            P0    = -rho_0 * phi_0; % pressure scale @ r_c

            % r0 = r_c              % LENGTH UNIT
            m0    = rho_0 * r_c^3;  % MASS UNIT
            t0    = r_c / v0;       % TIME UNIT
            %--- derived:
            u0    = m0 * v0^2;      % ENERGY UNIT

            nt = self.temperatureExponent;
            np = self.densityExponent;
            alpha = 1 + self.dustFraction; % inertial/gravitational density enhancement factor

            % scale height as a function of cylindrical radius
%            scaleHeight = cs_0 *radpts.^(1.5+.5*nt) / sqrt(GM);
            
            % calculate density at midplane
            q      = (-nt - 3 + 2*np)/2;
            rho_mp = rho_0 * radpts.^q;

            % Calculate rho at all elevations given T(r,z) = T(r,z=0) (vertically isothermal)
            % solve dP/dz = T drho/dz = -dphi/dzi
            %deltaphi = -phi_0*(1./sqrt(radpts.^2 + zpts.^2) - 1./radpts);
            deltaphi = -phi_0*(1./rsph - 1./radpts);
            mass = rho_mp .* exp(deltaphi .* radpts.^(-nt) / cs_0^2);
clear rho_mp; % never used again
            % Still more or less fudging the thermodynamics for now
            % note that throughout here, kb/mu is flagrantly absorbed onto temp/pressure
            self.fluidDetails(1) = fluidDetailModel('warm_molecular_hydrogen'); 
            self.fluidDetails(1).minMass = mpi_max(max(mass(:))) * self.densityCutoffFraction;

            vel     = geo.zerosXYZ(geo.VECTOR);

            if self.gasPerturb > 0
                vel(1,:,:,:) = self.gasPerturb * cs_0 * (geo.randsXYZ(geo.SCALAR)-.5) .*  (mass > self.fluidDetails(1).minMass);
                vel(2,:,:,:) = self.gasPerturb * cs_0 * (geo.randsXYZ(geo.SCALAR)-.5) .*  (mass > self.fluidDetails(1).minMass);
                vel(3,:,:,:) = self.gasPerturb * cs_0 * (geo.randsXYZ(geo.SCALAR)-.5) .*  (mass > self.fluidDetails(1).minMass);
            end

            rez = geo.globalDomainRez;
            [clipx, clipy, clipz] = geo.toLocalIndices([1:4 (rez(1)-3):rez(1)], [1:4 (rez(2)-3):rez(2)], [1:4 (rez(3)-3):rez(3)]);
            % no v perturbation at x-/x+ limits
            vel(:,clipx,:,:) = 0;
            % no v perturbation at z-/z+ limits
            if rez(3) > 1
                vel(:,:,:,clipz) = 0;
            end

            % Calculate the orbital velocity that solves radial force balance
            % Rotation has a small degree of vertical shear
            % Solution to v^2 / x = GM x^2/r^3 + (dP/dx)/rho
            vel(2,:,:,:) = squish(vel(2,:,:,:),'onlyleading') + sqrt( alpha*cs_0^2*(q+nt)*radpts.^(nt) - phi_0*( (nt+1)./radpts - nt./rsph) ); 

            % Setup internal energy density to yield correct P for an adiabatic ideal gas
            Eint = cs_0^2 * mass .* radpts.^nt / (self.fluidDetails(1).gamma-1);

            % no further references to radial points array 
            clear radpts;
            
            % Rescale if we choose to
            if self.normalizeValues
                mass = mass / rho_0;
                vel = vel / v0;
                Eint = Eint / P0;
                nfact = rho_0 / P0;
            else
                nfact = 1;
            end

            fluids(1) = self.rhoVelEintToFluid(mass, vel, Eint);

            if self.dustFraction > 0
                %uniformly disperse dust mass as a fixed fraction of the gas mass
                mass = mass * self.dustFraction;
                self.fluidDetails(2) = fluidDetailModel('10um_iron_balls');
                self.fluidDetails(2).minMass = mpi_max(max(mass(:))) * self.densityCutoffFraction;

                dscale = 30;
                SaveManager.logPrint('WARNING: Dust dynamics being setup with hack. Dynamics are those of %.3fmm iron spheres.\n', dscale / 100);    
                self.fluidDetails(2).sigma = self.fluidDetails(2).sigma * dscale^2;
                self.fluidDetails(2).mass = self.fluidDetails(2).mass * dscale^3;

                % Assert randomly generated perturbation velocity to dust where gas rho > value
                if self.dustPerturb > 0
                    SaveManager.logPrint('WARNING: run.dustPerturb is set, but independent gas/dust velocity perturbations are not supported: Set .gasPerturb instead.\n');
                end
                
                Eint = nfact*mass * (.01*cs_0)^2 / ((self.fluidDetails(2).gamma - 1));
                fluids(2) = self.rhoVelEintToFluid(mass, vel, Eint);
            else
                clear mass;
                clear vel;
                clear Eint;
                self.numFluids = 1;
            end
            
            for n = 1:self.numFluids
                self.fluidDetails(n) = rescaleFluidDetails(self.fluidDetails(n), m0, r_c, t0);
            end

            % Compute frame boost that minimizes the average advection speed & so maximizes timestep
            velInner = sqrt(GM / self.innerRadius);
            velOuter = sqrt(GM / self.outerRadius);
            self.frameParameters.rotateCenter = [0 0 0];
            self.frameParameters.omega = (velInner + velOuter)/(self.innerRadius + self.outerRadius);
            if self.normalizeValues
                self.frameParameters.omega = self.frameParameters.omega / w0;
            end
            
            % run for 100 years (@ this orbital radius)
            self.timeMax = 100*2*pi/self.frameParameters.omega;

            mag     = [0;0;0]; %geo.zerosXYZ(geo.VECTOR);
                          
            statics = [];%StaticsInitializer(self.grid);

            selfGravity = [];
            %starX = (self.grid+1)*dGrid/2;
            %selfGravity = SelfGravityInitializer();
            %selfGravity.compactobjectStates = [1 self.pointRadius starX(1) starX(2) starX(3) 0 0 0 0 0 0];
                                           % [m R x y z vx vy vz lx ly lz]
            
            potentialField = PotentialFieldInitializer();

            if self.normalizeValues
                potentialField.field = -1./rsph;
                potentialField.constant = 1;
            else
                potentialField.field = -GM ./ rsph;
                potentialField.constant = 1;
            end

            % Constructs a single-parameter softened potential -sqrt(2)/sqrt(r^2 + r0^2) inside pointRadius to avoid
            % singularity at r -> 0
            %soft = (sphericalR < self.Rstar);
            %phiSoft = -sqrt(2./(sphericalR(soft).^2 + self.Rstar^2));
            
            %potentialField.field(soft) = phiSoft;         
        end
        
    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
