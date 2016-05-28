classdef SodShockSolution < handle
    % Class annotation template for creating new classes.
    
    
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %							C O N S T A N T	 [P]
        GAMMA       = 1.4;
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %							P U B L I C  [P]
        time;
        x;
        mass;
        soundSpeed;
        pressure;
        velocity;
        energy;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %				   P R O T E C T E D [P]
        pTubeLength;        % Length of the tube.                                       double
    end %PROTECTED
    
    
    
    %===================================================================================================
    methods %																	  G E T / S E T  [M]
        
        %___________________________________________________________________________________________________ SodShockSolution
        function obj = SodShockSolution(resolution, time)
            
            obj.time                    = time;
            g                           = obj.GAMMA;
            x0                          = floor(resolution/2);
            
            obj.mass                    = zeros(1, resolution);
            obj.mass(1:x0)              = 1;
            obj.mass(x0:end)            = 0.125;
            
            obj.pressure                = zeros(1, resolution);
            obj.pressure(1:x0)          = 1;
            obj.pressure(x0:end)        = 0.1;
            
            obj.velocity                = zeros(1, resolution);
            obj.soundSpeed              = sqrt(g*obj.pressure./obj.mass);
            
            %--- Post Shock ---%
            postPressure                = obj.findPostShockPressure();
            postVelocity                = obj.calculatePostShockVelocity(postPressure);
            postMass                    = obj.calculatePostShockMass(postPressure);
            
            shockSpeed                  = obj.calculateShockSpeed(postMass, postVelocity);
            postContactMass             = obj.calculatePostContactMass(postPressure);
            
            %--- Rarefaction ---%
            rarefactionSoundSpeed       = obj.calculateRarefactionSoundSpeed(0.5, time);
            rarefactionVelocity         = obj.calculateRarefactionVelocity(0.5, time);
            rarefactionMass             = obj.calculateRarefactionMass(rarefactionSoundSpeed);
            rarefactionPressure         = obj.calculateRarefactionPressure(rarefactionMass);
            
            %--- Find Positions ---%
            x1true                      = (0.5 - obj.soundSpeed(1)*time); % fan's head
            x2true = .5 - time*((g+1)/(g-1))*obj.soundSpeed(1)*( (postContactMass/obj.mass(1))^(.5*g-.5) - 2/(g+1));
            x3true                      = 0.5 + postVelocity*time; % material contact
            x4true                      = 0.5 + shockSpeed*time; % shock position
            
            % Cells' left edge exists at (N-1)h and right edge at Nh,
            % with h = 1/N and N indexing from 1 to resolution.
            % If we consist with the notion of x_initialcontact = 1/2.
            h = 1/resolution;
            posLeft = (0:(resolution-1))*h;
            posRite = (1:resolution)*h;
            posCtr  = (posLeft + posRite)/2;
            
            
            % Identify regions consisting entirely of a single function
            leftConstRegion   = (posLeft < x1true);
            fanRegion         = (posLeft > x1true) & (posRite < x2true);
            postContactRegion = (posLeft > x2true) & (posRite < x3true);
            preContactRegion  = (posLeft > x3true) & (posRite < x4true);
            preShockRegion    = (posLeft > x4true);
            
            % A function to find the cell in which x resides:
            cellStraddling = @(x) find((posLeft < x) & (posRite > x));
            
            % Plug in left
            obj.mass(leftConstRegion) = 1;
            
            % Handle [Uleft | Fan] cell
            id = cellStraddling(x1true);
            
            g = obj.GAMMA;
            G = (g-1)/(g+1);
            cRF = @(x) G*((0.5-x)/time) + (1-G)*obj.soundSpeed(1);
            
            % Compute the exact integral average over the cell where the rarefaction's head resides
            denfunc = @(x) obj.calculateRarefactionMass(cRF(x));
            obj.mass(id) = ( integral(denfunc, x1true, posRite(id)) + (x1true - posLeft(id))*1 ) / h;
            
            % Use cell center mass (accurate to 3rd order of smallness) in rarefaction region
            obj.mass(fanRegion) = denfunc(posCtr(fanRegion));
            
            % Identify fan-to-postcontact cell and average it
            id = cellStraddling(x2true);
            obj.mass(id) = ( integral(denfunc, posLeft(id), x2true) + (posRite(id) - x2true)*postContactMass ) / h;
            
            % Plug in post contact mass density as constant
            obj.mass(postContactRegion) = postContactMass;
            
            % Average two halves of the contact's cell
            id = cellStraddling(x3true);
            obj.mass(id) = ( postContactMass*(x3true-posLeft(id)) + postMass*(posRite(id)-x3true) ) / h;
            
            % Plug in precontact/postshock mass
            obj.mass(preContactRegion) = postMass;
            
            % Average the shock's cell
            id = cellStraddling(x4true);
            obj.mass(id) = ( postMass*(x4true-posLeft(id)) + .125*(posRite(id)-x4true) ) / h;
            
            % Plug in original right-state density
            obj.mass(preShockRegion) = .125;
            
            % PRESSURE CALCULATION
            
            % Left of contact is adiabatic:
            pressA = leftConstRegion | fanRegion | postContactRegion;
            obj.pressure(pressA) = 1 * (obj.mass(pressA) / 1).^obj.GAMMA; % FIXME: generalize this
            
            id = cellStraddling(x2true);
            obj.pressure(id) = obj.mass(id)^obj.GAMMA;
            
            id = cellStraddling(x3true);
            obj.pressure(id) = obj.pressure(id-1);
            
            % Right of contact is postshock
            obj.pressure(preContactRegion) = postPressure;
            
            % Before shock is undisturbed right state
            
            % VELOCITY CALCULATION
            % find speed in the fan region
            vf = postVelocity;
            obj.velocity(fanRegion) = vf * (posCtr(fanRegion) - x1true) / (x2true-x1true);
            
            obj.velocity(postContactRegion | preContactRegion) = postVelocity;
            
            id = cellStraddling(x2true);
            obj.velocity(id) = postVelocity;
            id = cellStraddling(x3true);
            obj.velocity(id) = postVelocity;
            
            obj.soundSpeed              = sqrt(g*obj.pressure./obj.mass);
            obj.energy                  = obj.pressure./(g - 1) ...
                + 0.5*obj.mass.*obj.velocity.^2;
        end
        
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %														P U B L I C  [M]
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %											P R O T E C T E D    [M]
        
        %___________________________________________________________________________________________________ findPostShockPressure
        function result = findPostShockPressure(obj)
            g = obj.GAMMA;
            G = (g-1)/(g+1);
            b = (g-1)/(2*g);
            
            m1 = obj.mass(1);
            m2 = obj.mass(end);
            p1 = obj.pressure(1);
            p2 = obj.pressure(end);
            
            func = @(p)...
                (p1^b - p^b)*sqrt((1-G^2)*p1^(1/g)/(G^2*m1)) ...
                - (p - p2)*sqrt((1-G)/(m2*(p+G*p2)));
            result = fzero(func, 0.3);
        end
        
        %___________________________________________________________________________________________________ calculatePostShockVelocity
        function result = calculatePostShockVelocity(obj, postPressure)
            g      = obj.GAMMA;
            result = 2*sqrt(g)/(g - 1)*(1 - postPressure^((g - 1)/(2*g)));
        end
        
        %___________________________________________________________________________________________________ calculatePostShockMass
        function result = calculatePostShockMass(obj, postPressure)
            g      = obj.GAMMA;
            G      = (g-1)/(g+1);
            m2     = obj.mass(end);
            p2     = obj.pressure(end);
            
            result = m2*((postPressure/p2) + G) / (1 + G*(postPressure/p2));
        end
        
        %___________________________________________________________________________________________________ calculateShockSpeed
        function result = calculateShockSpeed(obj, postMass, postVelocity)
            m2     = obj.mass(end);
            
            result = postVelocity*(postMass/m2) / ((postMass/m2) - 1);
        end
        
        %___________________________________________________________________________________________________ calculatePostContactMass
        function result = calculatePostContactMass(obj, postPressure)
            g      = obj.GAMMA;
            p1     = obj.pressure(1);
            
            result = obj.mass(1)*(postPressure/p1)^(1/g);
        end
        
        %___________________________________________________________________________________________________ calculateRarefactionSoundSpeed
        function result = calculateRarefactionSoundSpeed(obj, x0, time)
            g         = obj.GAMMA;
            G         = (g-1)/(g+1);
            c1        = obj.soundSpeed(1);
            positions = linspace(0, 1, length(obj.mass));
            
            result    = G*((x0-positions)./time) + (1 - G)*c1;
        end
        
        %___________________________________________________________________________________________________ calculateRarefactionVelocity
        function result = calculateRarefactionVelocity(obj, x0, time)
            g         = obj.GAMMA;
            G         = (g-1)/(g+1);
            c1        = obj.soundSpeed(1);
            positions = linspace(0, 1, length(obj.soundSpeed));
            
            result    = (1 - G)*((positions-x0)/time + c1);
        end
        
        %___________________________________________________________________________________________________ calculateRarefactionMass
        function result = calculateRarefactionMass(obj, rarefactionSoundSpeed)
            g      = obj.GAMMA;
            m1     = obj.mass(1);
            c1     = obj.soundSpeed(1);
            
            result = m1.*(rarefactionSoundSpeed./c1).^(2/(g-1));
        end
        
        %___________________________________________________________________________________________________ calculateRarefactionPressure
        function result = calculateRarefactionPressure(obj, rarefactionMass)
            g      = obj.GAMMA;
            m1     = obj.mass(1);
            p1     = obj.pressure(1);
            
            result = p1*(rarefactionMass./m1).^g;
        end
        
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %													  S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
