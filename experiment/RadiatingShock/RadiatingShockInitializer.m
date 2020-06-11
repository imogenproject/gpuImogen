classdef RadiatingShockInitializer < Initializer
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
        RANDOM          = 'random';
        COSINE          = 'cosine';
        COSINE_2D       = 'cosine_2d';
    end%CONSTANT
    
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
        perturbationType; % Enumerated type of perturbation used to seed.             str
        seedAmplitude;    % Maximum amplitude of the seed noise values.               double
        cos2DFrequency;   % Resolution independent frequency for the cosine 2D        double
                          %   perturbations in both y and z.
        randomSeed_spectrumLimit; % Max ky/kz mode# seeded by a random perturbation

        theta;
        sonicMach;
        alfvenMach; % Set to -1 for hydrodynamic

        machY_boost; % Galiean transform makes everything slide sideways at this mach;
                     % Prevent HLLC from being a victim of its own success
        fallbackBoost;

        radBeta;          % Radiation rate = radBeta P^radTheta rho^(2-radTheta)
        radTheta; 

        fractionPreshock; % The fraction of the grid to give to preshock cells; [0, 1)
        fractionCold;     % The fraction of the grid to give to the cold gas layer; [0, 1)
        Tcutoff; % Forces radiation to stop at this temperature; Normalized by preshock gas temp.
                            % Thus we have [nx*f = preshock | nx*(1-f) = postshock] with hx = len_singularity * f_t / (nx*(1-f));
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%_________________________________________________________________________ RadiatingShockInitializer
% Creates an Iiitializer for corrugation shock simulations. Takes a single input argument that is
% either the size of the grid, e.g. [300, 6, 6], or the full path to an existing results data file
% to use in loading
        function obj = RadiatingShockInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'RADHD';
            obj.info             = 'Radiating HD shock';
            obj.fractionPreshock = .25;
            obj.fractionCold     = .1;
            obj.Tcutoff          = 1.05;
            % This is actually P2/rho2 == Tcutoff*(P1/rho1)

            obj.mode.fluid       = true;
            obj.mode.magnet      = false;
            obj.mode.gravity     = false;
            obj.treadmill        = false;
            obj.iterMax          = 100;
            obj.bcMode.x         = ENUM.BCMODE_CONSTANT;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            obj.activeSlices.xy  = true;
            obj.activeSlices.xyz = true;
            obj.ppSave.dim2      = 5;
            obj.ppSave.dim3      = 25;
            obj.image.mass       = false;
            obj.image.interval   = 10;

            obj.perturbationType = RadiatingShockInitializer.RANDOM;
            obj.randomSeed_spectrumLimit = 64; % 
            obj.seedAmplitude    = 5e-4;

           
            obj.theta            = 0;
            obj.sonicMach        = 3;
            obj.alfvenMach       = -1;

            obj.machY_boost      = .02*obj.sonicMach;
            obj.fallbackBoost    = 0; 

            obj.radTheta = .5;
            obj.radBeta = 1;
            
            obj.logProperties    = [obj.logProperties, 'gamma'];

            obj.operateOnInput(input, [512 1 1]);
            obj.pureHydro = 1;
        end

    end%GET/SET
    
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
        
    end%PUBLIC
    
%===================================================================================================    
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
%________________________________________________________________________ calculateInitialConditions
        function [fluids, mag, statics, potentialField, selfGravity] = calculateInitialConditions(obj)
            % Returns the initial conditions for a corrugation shock wave according to the settings for
            % the initializer.
            % USAGE: [mass, mom, ener, mag, statics, run] = getInitialConditions();
            potentialField = [];
            selfGravity = [];
            geom = obj.geomgr;
            geom.makeDimNotCircular(1);
            
            rad = RadiationSubInitializer();

            rad.type                      = ENUM.RADIATION_OPTICALLY_THIN;
            rad.exponent                  = obj.radTheta;

            rad.initialMaximum            = 1; % We do not use these, instead
            rad.coolLength                = 1; % We let the cooling function define dx
            rad.strengthMethod            = 'preset';
            rad.setStrength               = obj.radBeta;
            
            statics = []; % We'll set this eventually...
            
            % Gets the jump solution, i.e. preshock and adiabatic postshock solutions
            if obj.alfvenMach < 0
                jump = HDJumpSolver (obj.sonicMach, obj.theta, obj.gamma);
                obj.pureHydro = 1;
            else
                jump = MHDJumpSolver(obj.sonicMach, obj.alfvenMach, obj.theta, obj.gamma);
                obj.pureHydro = 0;
                obj.mode.magnet = true;
            end
            radflow = RadiatingFlowSolver(jump.rho(2), jump.v(1,2), ...
                jump.v(2,2), jump.B(1,2), jump.B(2,2), ...
                jump.Pgas(2), obj.gamma, obj.radBeta, obj.radTheta, 0);
            
            radflow.numericalSetup(2, 2);
            
            L_c = radflow.coolingLength(jump.v(1,2));
            T_c = radflow.coolingTime(jump.v(1,2));
            
            radflow.setCutoff('thermal',obj.Tcutoff);
            
            flowEndpoint = radflow.calculateFlowTable(jump.v(1,2), L_c / 1000, 5*L_c);
            flowValues   = radflow.solutionTable();
            
            SaveManager.logPrint('Characteristic cooling length: %f\nCharacteristic cooling time:   %f\nDistance to singularity: %f\n', L_c, T_c, flowEndpoint);
            
            fracFlow = 1.0-(obj.fractionPreshock + obj.fractionCold);
            
            geom.makeBoxSize(flowEndpoint / fracFlow); % Box length
            
            [vecX, vecY, vecZ] = geom.ndgridVecs();
            
            rez = geom.globalDomainRez;
            
            if 1 % if initializing analytically
                % Identify the preshock, radiating and cold gas layers.
                preshock =  (vecX < rez(1)*obj.fractionPreshock);
                postshock = (vecX >= rez(1)*obj.fractionPreshock);
                postshock = postshock & (vecX < rez(1)*(1-obj.fractionCold));
                coldlayer = (vecX >= rez(1)*(1-obj.fractionCold));
                
                numPre = rez(1)*obj.fractionPreshock;
                Xshock = geom.d3h(1)*numPre;
                
                % Generate blank slates
                [mass, mom, mag, ener] = geom.basicFluidXYZ();
                
                % Fill in preshock values of uniform flow
                mass(preshock,:,:) = jump.rho(1);
                ener(preshock,:,:) = jump.Pgas(1);
                mom(1,:,:,:) = jump.rho(1)*jump.v(1,1);
                mom(2,preshock,:,:) = jump.rho(1)*jump.v(2,1);
                mag(1,:,:,:) = jump.B(1,1);
                mag(2,preshock,:,:) = jump.B(2,1);
                
                
                % Get interpolated values for the flow
                flowValues(:,1) = flowValues(:,1) + Xshock;
                xinterps = (vecX-.5)*geom.d3h(1);
                
                meth = 'spline';
                minterp = interp1(flowValues(:,1), flowValues(:,2), xinterps, meth);
                %px is a preserved invariant
                pyinterp= interp1(flowValues(:,1), flowValues(:,2).*flowValues(:,4), xinterps, meth);
                %bx is a preserved invariant
                byinterp= interp1(flowValues(:,1), flowValues(:,6), xinterps, meth);
                Pinterp = interp1(flowValues(:,1), flowValues(:,7), xinterps, meth);
                
                for xp = find(postshock)
                    mass(xp,:,:) = minterp(xp);
                    mom(2,xp,:,:) = pyinterp(xp);
                    mag(2,xp,:,:) = byinterp(xp);
                    ener(xp,:,:)  = Pinterp(xp);
                end
                
                % Fill in cold gas layer adiabatic values again
                endstate = flowValues(end,:);
                
                mass(coldlayer,:,:) = endstate(2);
                % px is constant
                mom(2,coldlayer,:,:) = endstate(2)*endstate(4);
                % bx is constant
                mag(2,coldlayer,:,:) = endstate(6);
                ener(coldlayer,:,:) = endstate(7);
            
            else
                % load the file
                %F = DataFrame( . );
                
                xinput = (1:size(F.mass,1)) * F.dGrid{1};
                dx = F.dGrid{1};

                % These return physical positions
                topcell = trackFront2(F.mass, xinput, 2);
                botcell = RHD_utils.trackColdBoundary(F);
                
                ra = round(topcell / dx - 10);
                rb = round(botcell / dx + 10);
                
                origBoost = F.velX(1,1) - jump.v(1,1);
                
                
                
                
            end
            
            
            %----------- BOOST TRANSFORM ---------%
            dvy = jump.v(1,1)*obj.machY_boost/obj.sonicMach;
            yboost = mass*dvy;
            mom(2,:,:,:) = squish(mom(2,:,:,:)) + yboost;
            if obj.fallbackBoost ~= 0
                xboost = mass*obj.fallbackBoost;
                mom(1,:,:,:) = squish(mom(1,:,:,:)) - xboost;
            end
            
            %----------- SALT TO SEED INSTABILITIES -----------%
            % Salt everything from half the preshock region to half the cooling region.
            fracRadiate = 1.0-obj.fractionPreshock-obj.fractionCold;
            Xsalt = (vecX >= (.5*rez(1)*obj.fractionPreshock)) & (vecX < rez(1)*(obj.fractionPreshock + .5*fracRadiate));
            
            switch (obj.perturbationType)
                % RANDOM Seeds ____________________________________________________________________
                case RadiatingShockInitializer.RANDOM
                    %phase = 2*pi*rand(10,obj.grid(2), obj.grid(3));
                    %amp   = obj.seedAmplitude*ones(1,obj.grid(2), obj.grid(3))*obj.grid(2)*obj.grid(3);
                    
                    %amp(:,max(4, obj.randomSeed_spectrumLimit):end,:) = 0;
                    %amp(:,:,max(4, obj.randomSeed_spectrumLimit):end) = 0;
                    %amp(:,1,1) = 0; % no common-mode seed
                    
                    %perturb = zeros(10, obj.grid(2), obj.grid(3));
                    %for xp = 1:size(perturb,1)
                    %    perturb(xp,:,:) = sin(xp*2*pi/20)^2 * real(ifft(squish(amp(1,:,:).*exp(1i*phase(1,:,:)))));
                    %end
                    junk = obj.seedAmplitude*(rand(size(mass))-.5);
                    mass(Xsalt,:,:) = mass(Xsalt,:,:) + junk(Xsalt,:,:);
                case RadiatingShockInitializer.COSINE
                    [X, Y, Z] = ndgrid(1:delta, 1:obj.grid(2), 1:obj.grid(3));
                    perturb = obj.seedAmplitude*cos(2*pi*(Y - 1)/(obj.grid(2) - 1)) ...
                        .*sin(pi*(X - 1)/(delta - 1));
                    % COSINE Seeds ____________________________________________________________________
                case RadiatingShockInitializer.COSINE_2D
                    [X, Y, Z] = ndgrid(1:delta, 1:obj.grid(2), 1:obj.grid(3));
                    perturb = obj.seedAmplitude ...
                        *( cos(2*pi*obj.cos2DFrequency*(Y - 1)/(obj.grid(2) - 1)) ...
                        + cos(2*pi*obj.cos2DFrequency*(Z - 1)/(obj.grid(3) - 1)) ) ...
                        .*sin(pi*(X - 1)/(delta - 1));
                    
                    % Unknown Seeds ___________________________________________________________________
                otherwise
                    error('Imogen:CorrugationShockInitializer', ...
                        'Uknown perturbation type. Aborted run.');
            end
            
            %            seeds = seedIndices(mine);
            %
            %            mass(seedIndices,:,:) = squish( mass(seedIndices,:,:) ) + perturb;
            % Add seed to mass while maintaining self-consistent momentum/energy
            % This will otherwise take a dump on e.g. a theta=0 shock with
            % a large density fluctuation resulting in negative internal energy
            %            mom(1,seedIndices,:,:) = squish(mass(seedIndices,:,:)) * jump.v(1,1);
            %            mom(2,seedIndices,:,:) = squish(mass(seedIndices,:,:)) * jump.v(2,1);
            
            ener = ener/(obj.gamma-1) + ...
                .5*squish(sum(mom.^2,1))./mass + .5*squish(sum(mag.^2,1));
            
            statics = StaticsInitializer(geom);
            
            %statics.setFluid_allConstantBC(mass, ener, mom, 1);
            %statics.setMag_allConstantBC(mag, 1);
            
            %statics.setFluid_allConstantBC(mass, ener, mom, 2);
            %statics.setMag_allConstantBC(mag, 2);
            
            
            fluids = obj.rhoMomEtotToFluid(mass, mom, ener);
            
            obj.radiation = rad;
        end


    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
