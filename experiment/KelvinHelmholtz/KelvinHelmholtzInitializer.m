classdef KelvinHelmholtzInitializer < Initializer

% Creates initial conditions for a Kelvin-Helmholtz instability, where two anti-parallel shearing 
% flows are seeded with noise along the shearing barrier. The perturbation of the barrier causes
% the well known Kelvin-Helmholtz instabiliity phenomena to form. It should be noted that to 
% prevent a large shock wave from forming across the barrier the two regions, while having differing
% mass density values, have equal pressures. This makes sense given the usual observation of this
% phenomena in clouds, where such conditions exist.

%===================================================================================================
    properties (Constant = true, Transient = true) %                            C O N S T A N T  [P]
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        mach;
        waveHeight;
        nx;
        massRatio;      % ratio of (low mass)/(high mass) for the flow regions.     double
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
        function self = KelvinHelmholtzInitializer(input)
            self = self@Initializer();
            self.gamma            = 5/3;
            self.runCode          = 'KelHelm';
            self.info             = 'Kelvin-Helmholtz instability test';
            self.pureHydro        = true;
            self.mode.fluid       = true;
            self.mode.magnet      = false;
            self.mode.gravity     = false;
            self.cfl              = 0.85;
            self.iterMax          = 1500;
            self.mach             = 0.25;
            self.activeSlices.xy  = true;
            self.ppSave.dim2      = 25;

            self.waveHeight       = .01;
            self.nx               = 1;

            self.bcMode.x         = 'circ';
            self.bcMode.y         = 'circ';
            self.bcMode.z         = 'mirror';
            
            self.massRatio        = 2;
            self.operateOnInput(input, [512 512 1]);

        end
        
    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]       
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%___________________________________________________________________________________________________ calculateInitialConditions
        function [fluid, mag, statics, potentialField, selfGravity] = calculateInitialConditions(self)
            
            %--- Initialization ---%
            statics             = [];
            potentialField      = [];
            selfGravity         = [];
            geo                 = self.geomgr;
            
            gdr = geo.globalDomainRez;
            if gdr(3) == 1; self.bcMode.z = 'circ'; end
            
            geo.makeBoxSize([1]);
            
            kx = 2*pi*self.nx;
            
            % The fluid shears at a rate of 2xSpeed,
            % with supper and slower computed to place us in the zero-momentum frame
            speed               = speedFromMach(self.mach, self.gamma, 1, 1/(self.gamma-1), 0);
            supper              =  speed - (self.massRatio-1)*speed/(self.massRatio+1);
            slower              = -speed - (self.massRatio-1)*speed/(self.massRatio+1);
            
            omega = -kx*sqrt(self.massRatio*(supper - slower)^2 / (1+self.massRatio)^2);
            
            % Initialize Arrays
            [mass, mom, mag, ener] = geo.basicFluidXYZ();
            
            % Initialize parallelized vectors
            [X, Y, Z]           = geo.ndgridSetIJK('pos');
            
            yctr = .5*gdr(2)/gdr(1);
            
            % Define the wave contact
            topwave             = yctr - self.waveHeight*cos(kx*X);
            heavy               = (Y > topwave);
            
            eta       = yctr + self.waveHeight * cos(kx*X);
            mass      = 1 + (self.massRatio - 1)*(Y > eta);
            
            vxupper   = supper + self.waveHeight*real(-(kx*supper - omega)*exp(1i*kx*X - kx*(Y-yctr)));
            vyupper   = self.waveHeight*real(1i*(kx*supper - omega)*exp(1i*kx*X - kx*(Y-yctr)));
            
            vx        = slower + self.waveHeight*real(-(kx*slower - omega)*exp(1i*kx*X + kx*(Y-yctr)));
            vy        = self.waveHeight*real(-1i*(kx*slower - omega)*exp(1i*kx*X + kx*(Y-yctr)));
            
            vx(heavy) = vxupper(heavy);
            vy(heavy) = vyupper(heavy);
            
            
            % Define properties of the slabs
            mass(heavy)  = self.massRatio;
            
            mom(1,:,:,:) = vx .* mass;
            mom(2,:,:,:) = vy .* mass;
            
            % Calculate energy density array
            ener = (maxFinderND(mass).^self.gamma)/(self.gamma - 1) ...             % internal
                + 0.5*squish(sum(mom.*mom,1))./mass ...                             % kinetic
                + 0.5*squish(sum(mag.*mag,1));                                      % magnetic
            
            fluid = self.rhoMomEtotToFluid(mass, mom, ener);
        end
    end%PROTECTED
    
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
