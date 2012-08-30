classdef CorrugationShockInitializer < Initializer
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
        perturbationType;   % Enumerated type of perturbation used to seed.             str
        seedAmplitude;      % Maximum amplitude of the seed noise values.               double
        cos2DFrequency;     % Resolution independent frequency for the cosine 2D        double
                            %   perturbations in both y and z.
        randomSeed_spectrumLimit; % Max ky/kz mode# seeded by a random perturbation

        theta;
        sonicMach;
        alfvenMach;

        numericalICfile    % File contains previous run of these ICs
        endMass; % if not zero, sets mass at +x end to this (for radiative shocks 1.0)
    end %PUBLIC

%===================================================================================================
    properties (Dependent = true) %                                            D E P E N D E N T [P]
        dataFile;
    end %DEPENDENT
    
%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        mass;               % pre and post shock mass density values.                   double(2)
        velocity;           % pre and post shock momentum density values.               double(3,2)
        pressure;           % pre and post shock pressure density values.               double(2)
        magnet;             % pre and post shock magnetic field density values.         double(3,2)
    end %PROTECTED
    
    
    
    
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        
%___________________________________________________________________________________________________ CorrugationShockInitializer
% Creates an Iiitializer for corrugation shock simulations. Takes a single input argument that is
% either the size of the grid, e.g. [300, 6, 6], or the full path to an existing results data file
% to use in loading
        function obj = CorrugationShockInitializer(input)
            obj                  = obj@Initializer();
            obj.gamma            = 5/3;
            obj.runCode          = 'CORR';
            obj.info             = 'Corrugation shock trial.';
            obj.mode.fluid       = true;
            obj.mode.magnet      = true;
            obj.mode.gravity	 = false;
            obj.treadmill        = false;
            obj.cfl              = 0.35;
            obj.iterMax          = 10;
            obj.bcMode.x         = ENUM.BCMODE_FADE;
            obj.bcMode.y         = ENUM.BCMODE_CIRCULAR;
            obj.bcMode.z         = ENUM.BCMODE_CIRCULAR;
            obj.bcInfinity       = 20;
            obj.activeSlices.xy  = true;
            obj.activeSlices.xyz = true;
            obj.ppSave.dim2      = 5;
            obj.ppSave.dim3      = 25;
            obj.image.mass       = true;
            obj.image.interval   = 10;

%            obj.dGrid.x.points   = [0, 5;    33.3, 1;    66.6, 1;    100, 5];
            
            obj.perturbationType = CorrugationShockInitializer.RANDOM;
            obj.randomSeed_spectrumLimit = 64; % 
            obj.seedAmplitude    = 5e-3;
            
            obj.theta            = 10;
            obj.sonicMach        = 10;
            obj.alfvenMach       = 0.125;
            
            obj.logProperties    = [obj.logProperties, 'gamma'];

            obj.numericalICfile  = 'null';
            obj.endMass = 0;
            
            obj.operateOnInput(input, [300, 6, 6]);
        end

%___________________________________________________________________________________________________ dataFile
        function result = get.dataFile(obj)
            gammaParameter = '';
            if obj.gamma ~= 5/3
                gammaParameter = ['-g', strrep(num2str(obj.gamma, '%0.3g'), '.', '_')];
            end
            
            result = sprintf('ics-t%s-ms%s-ma%s%s.mat', ...
                            strrep(num2str(obj.theta,'%0.3g'),'.','_'), ...
                            strrep(num2str(obj.sonicMach,'%0.3g'),'.','_'), ...
                            strrep(num2str(obj.alfvenMach,'%g'),'.','_'), ...
                            gammaParameter);
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


        %--- Attempt to load data from file ---%
            result = [];
            relaxed = [];
            
            try
                disp('Attempting to access shock database...'); 
                mksqlite(1,'open','experiment/experimentUtils/mhd_jumpIC.sqlite.db');

                result = mksqlite(1,sprintf('select * from main.analytical where ms=%i and ma=%i and theta=%i', ...
                    round(100*obj.sonicMach), round(100*obj.alfvenMach),obj.theta));
                if isempty(result); disp('No analytical result available in DB; Solving jump analytically'); end

                relaxed = mksqlite(1,sprintf('select * from main.numerical where ms=%i and ma=%i and theta=%i', ...
                    round(100*obj.sonicMach),round(100*obj.alfvenMach),obj.theta));
                if isempty(relaxed); disp('Pre-relaxed numerical curve not available; Shock will generate transient.'); end

                mksqlite(1,'close');
                
            catch didntwork
                disp('SQLite not available; Solving jump conditions analytically, no pre-relaxation');
            end
                
             if isempty(result)
                 result = MHDJumpSolver(obj.sonicMach, obj.alfvenMach, obj.theta, obj.gamma);
                 obj.mass       = result.mass;
                 obj.pressure   = result.pressure;
                 obj.velocity   = result.velocity;
                 obj.magnet     = result.magnet;
                 obj.theta      = result.theta;
                 obj.sonicMach  = result.sonicMach;
                 obj.alfvenMach = result.alfvenMach;
             else
                 obj.mass = [1; result.rho];
                 obj.pressure = [1; result.P2];
                 obj.velocity = [result.vx1 result.vx2; result.vy1 result.vy2; 0 0];
                 obj.magnet = [result.bx result.bx; result.by1 result.by2; 0 0];
                 obj.theta = result.theta;
                 obj.sonicMach = .01*result.ms;
                 obj.alfvenMach = .01*result.ma;
             end

            %--- Initialization ---%
            statics                 = []; % No statics used in this problem
            obj.dGrid.value         = 0.01/min(obj.grid(2:3));
            if obj.grid(3) == 1; obj.dGrid.value = .01/obj.grid(2); end
            obj.dGrid = obj.dGrid.value * ones(1,3);
            %obj.appendInfo('Grid cell spacing set to %g.',obj.dGrid.value);
            
            half        = ceil(obj.grid/2);
            preIndeces  = { 1:(half(1)-1), 1:obj.grid(2), 1:obj.grid(3) };
            postIndeces = { half(1):obj.grid(1), 1:obj.grid(2), 1:obj.grid(3) };
            
            %--- Create and populate data arrays ---%
            mass                 = zeros(obj.grid);
            mass(preIndeces{:})  = obj.mass(1);
            mass(postIndeces{:}) = obj.mass(2);
            
            mom                  = zeros([3 obj.grid]);
            mag                  = zeros([3 obj.grid]);
            for i=1:3
                mom(i,preIndeces{:})  = obj.velocity(i,1);
                mom(i,postIndeces{:}) = obj.velocity(i,2);
                mom(i,:,:,:)          = mass .* squeeze(mom(i,:,:,:));
                
                % It is assumed that the equations used to solve for the magnetic field already
                % include the transformation B = B/sqrt(4*pi) as verified by no 4*pi factors
                % existing in the original equations.
                mag(i,preIndeces{:})  = obj.magnet(i,1);
                mag(i,postIndeces{:}) = obj.magnet(i,2);
            end
            
            ener                 = zeros(obj.grid);
            ener(preIndeces{:})  = obj.pressure(1)/(obj.gamma - 1);     % internal energy (pre)
            ener(postIndeces{:}) = obj.pressure(2)/(obj.gamma - 1);     % internal energy (post)
            ener                 = ener ...
                                 + 0.5*squeeze( sum(mom.*mom,1) )./mass ...      % kinetic energy
                                 + 0.5*squeeze( sum(mag.*mag,1) );               % magnetic energy

            if ~isempty(relaxed)
                x0 = round(obj.grid(1)/2) - 16;
                relaxed.rho = eval(relaxed.rho);
                relaxed.px = eval(relaxed.px);
                relaxed.py = eval(relaxed.py);
                relaxed.bx = eval(relaxed.bx);
                relaxed.by = eval(relaxed.by);
                relaxed.E  = eval(relaxed.E);

                for c = 1:32; 
                    mass(x0+c,:,:)  = relaxed.rho(c);
                    ener(x0+c,:,:)  = relaxed.E(c);

                    mom(1,x0+c,:,:) = relaxed.px(c);
                    mom(2,x0+c,:,:) = relaxed.py(c);

                    mag(1,x0+c,:,:) = relaxed.bx(c);
                    mag(2,x0+c,:,:) = relaxed.by(c);
                end
            end
 
            %--- Perturb mass density in pre-shock region ---%
            %       Mass density gets perturbed in the pre-shock region just before the shock front
            %       to seed the formation of the instability.
            delta       = ceil(0.12*obj.grid(1));
            seedIndices = { (half(1)-20):(half(1)-11), 1:obj.grid(2), 1:obj.grid(3) };
            
            switch (obj.perturbationType)
                
                % RANDOM Seeds ____________________________________________________________________
                case CorrugationShockInitializer.RANDOM
                    phase = 2*pi*rand(10,obj.grid(2), obj.grid(3));
                    amp   = obj.seedAmplitude*ones(1,obj.grid(2), obj.grid(3))*obj.grid(2)*obj.grid(3);

                    amp(:,max(4, obj.randomSeed_spectrumLimit):end,:) = 0;
                    amp(:,:,max(4, obj.randomSeed_spectrumLimit):end) = 0;
                    amp(:,1,1) = 0; % no common-mode seed

                    perturb = zeros(10, obj.grid(2), obj.grid(3));
                    for xp = 1:size(perturb,1)
                        perturb(xp,:,:) = sin(xp*2*pi/20)^2 * real(ifft(squeeze(amp(1,:,:).*exp(1i*phase(1,:,:)))));
                    end


                case CorrugationShockInitializer.COSINE
                    [X Y Z] = ndgrid(1:delta, 1:obj.grid(2), 1:obj.grid(3));
                    perturb = obj.seedAmplitude*cos(2*pi*(Y - 1)/(obj.grid(2) - 1)) ...
                                    .*sin(pi*(X - 1)/(delta - 1));
                % COSINE Seeds ____________________________________________________________________
                case CorrugationShockInitializer.COSINE_2D 
                    [X Y Z] = ndgrid(1:delta, 1:obj.grid(2), 1:obj.grid(3));
                    perturb = obj.seedAmplitude ...
                                *( cos(2*pi*obj.cos2DFrequency*(Y - 1)/(obj.grid(2) - 1)) ...
                                 + cos(2*pi*obj.cos2DFrequency*(Z - 1)/(obj.grid(3) - 1)) ) ...
                                 .*sin(pi*(X - 1)/(delta - 1));

                % Unknown Seeds ___________________________________________________________________
                otherwise
                    error('Imogen:CorrugationShockInitializer', ...
                          'Uknown perturbation type. Aborted run.');
            end

            % By not perturbing energy density we assert that this is an entropy wave
            mass(seedIndices{:}) = squeeze( mass(seedIndices{:}) ) + perturb; % Add seed to mass.
            for i = 1:3; 
                % Maintain zero velocity perturbation
                mom(i,seedIndices{:}) = squeeze(mom(i,seedIndices{:})) + perturb * obj.velocity(i,1);
            end
        
            if obj.useGPU == true
                statics = StaticsInitializer(obj.grid); 

                %statics.setFluid_allConstantBC(mass, ener, mom, 1);
                %statics.setMag_allConstantBC(mag, 1);

                %statics.setFluid_allConstantBC(mass, ener, mom, 2);
                %statics.setMag_allConstantBC(mag, 2);
            end

            if obj.endMass > 0
                mass((end-10):end,:,:) = obj.endMass;
            end

        end


    end%PROTECTED
        
%===================================================================================================    
    methods (Static = true) %                                                     S T A T I C    [M]
    end
end%CLASS
