classdef SodShockSolution < handle
    % Class annotation template for creating new classes.
    
    
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                         C O N S T A N T         [P]
        GAMMA       = 1.4;
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                               P U B L I C  [P]
        time;
        x;
        mass;
        soundSpeed;
        pressure;
        velocity;
        energy;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                    P R O T E C T E D [P]
        pTubeLength;        % Length of the tube.                                       double
    end %PROTECTED
    
    
    
    %===================================================================================================
    methods %                                                                         G E T / S E T  [M]
        
        %______________________________________________________________________________ SodShockSolution
        function obj = SodShockSolution(xcoords, time)
            
            rez = size(xcoords);

            obj.time                    = time;
            g                           = obj.GAMMA;
            
            obj.mass                    = zeros(rez);
            obj.mass(xcoords < 0.0)     = 1;
            obj.mass(xcoords >= 0.0)    = 0.125;
            
            obj.pressure                = zeros(rez);
            obj.pressure(xcoords < 0.0) = 1;
            obj.pressure(xcoords >= 0.0)= 0.1;
            
            obj.velocity                = zeros(rez);
            obj.soundSpeed              = sqrt(g*obj.pressure./obj.mass);
            
            %--- Post Shock ---%
            postPressure                = obj.findPostShockPressure();
            postVelocity                = obj.calculatePostShockVelocity(postPressure);
            postMass                    = obj.calculatePostShockMass(postPressure);
            
            shockSpeed                  = obj.calculateShockSpeed(postMass, postVelocity);
            postContactMass             = obj.calculatePostContactMass(postPressure);
            
            %--- Rarefaction ---%
            %rarefactionSoundSpeed       = obj.calculateRarefactionSoundSpeed(0.5, time);
            %rarefactionVelocity         = obj.calculateRarefactionVelocity(0.5, time);
            %rarefactionMass             = obj.calculateRarefactionMass(rarefactionSoundSpeed);
            %rarefactionPressure         = obj.calculateRarefactionPressure(rarefactionMass);
            
            %--- Find Positions ---%
            xoffset = 0;
            x1true                      = (xoffset - obj.soundSpeed(1)*time); % fan's head
            x2true = xoffset - time*((g+1)/(g-1))*obj.soundSpeed(1)*( (postContactMass/obj.mass(1))^(.5*g-.5) - 2/(g+1));
            x3true                      = xoffset + postVelocity*time; % material contact
            x4true                      = xoffset + shockSpeed*time; % shock position
            
            % Cells' left edge exists at (N-1)h and right edge at Nh,
            % with h = 1/N and N indexing from 1 to resolution.
            % If we consist with the notion of x_initialcontact = 1/2.
            h = xcoords(2) - xcoords(1); % FIXME nasty hack

            posLeft = xcoords - h/2;
            posRite = xcoords + h/2;
            posCtr  = xcoords;
            
            
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
            cRF = @(x) G*((0-x)/time) + (1-G)*obj.soundSpeed(1);
            
            % Compute the exact integral average over the cell where the rarefaction's head resides
            denfunc = @(x) obj.calculateRarefactionMass(cRF(x));

            if ~isempty(id)
                obj.mass(id) = ( integral(denfunc, x1true, posRite(id)) + (x1true - posLeft(id))*1 ) / h;
            end

            % Use cell center mass (accurate to 3rd order of smallness) in rarefaction region
            obj.mass(fanRegion) = denfunc(posCtr(fanRegion));
            
            % Identify fan-to-postcontact cell and average it
            id = cellStraddling(x2true);
            if ~isempty(id)
               obj.mass(id) = ( integral(denfunc, posLeft(id), x2true) + (posRite(id) - x2true)*postContactMass ) / h;
            end
            
            % Plug in post contact mass density as constant
            obj.mass(postContactRegion) = postContactMass;
            
            % Average two halves of the contact's cell
            id = cellStraddling(x3true);
            if ~isempty(id)
                obj.mass(id) = ( postContactMass*(x3true-posLeft(id)) + postMass*(posRite(id)-x3true) ) / h;
            end
            
            % Plug in precontact/postshock mass
            obj.mass(preContactRegion) = postMass;
            
            % Average the shock's cell
            id = cellStraddling(x4true);
            if ~isempty(id)
                obj.mass(id) = ( postMass*(x4true-posLeft(id)) + .125*(posRite(id)-x4true) ) / h;
            end
            
            % Plug in original right-state density
            obj.mass(preShockRegion) = .125;
            
            % PRESSURE CALCULATION
            
            % Left of contact is adiabatic:
            pressA = leftConstRegion | fanRegion | postContactRegion;
            obj.pressure(pressA) = 1 * (obj.mass(pressA) / 1).^obj.GAMMA; % FIXME: generalize this
            
            id = cellStraddling(x2true);
            if ~isempty(id)
                obj.pressure(id) = obj.mass(id)^obj.GAMMA;
            end
            
            id = cellStraddling(x3true);
            if ~isempty(id)
                 obj.pressure(id) = obj.pressure(id-1);
            end
            
            % Right of contact is postshock
            obj.pressure(preContactRegion) = postPressure;
            
            % Before shock is undisturbed right state
            
            % VELOCITY CALCULATION
            % find speed in the fan region
            vf = postVelocity;
            obj.velocity(fanRegion) = vf * (posCtr(fanRegion) - x1true) / (x2true-x1true);
            
            obj.velocity(postContactRegion | preContactRegion) = postVelocity;
            
            id = cellStraddling(x2true);
            if ~isempty(id)
                obj.velocity(id) = postVelocity;
            end
            id = cellStraddling(x3true);
            if ~isempty(id)
                obj.velocity(id) = postVelocity;
            end

            obj.soundSpeed              = sqrt(g*obj.pressure./obj.mass);
            obj.energy                  = obj.pressure./(g - 1) ...
                + 0.5*obj.mass.*obj.velocity.^2;
        end
        
    end%GET/SET
    
    %===============================================================================================
    methods (Access = public) %                                                                                                                P U B L I C  [M]
    end%PUBLIC
    
    %===============================================================================================
    methods (Access = protected) %                                          P R O T E C T E D    [M]
        
        %_____________________________________________________________________ findPostShockPressure
        function result = findPostShockPressure(obj)
            g = obj.GAMMA;
            G = (g-1)/(g+1);
            b = (g-1)/(2*g);
            
            m1 = 1; % FIXME: don't hardcode this crap FFS
            m2 = .125;
            p1 = 1;
            p2 = .1;
            
            func = @(p)...
                (p1^b - p^b)*sqrt((1-G^2)*p1^(1/g)/(G^2*m1)) ...
                - (p - p2)*sqrt((1-G)/(m2*(p+G*p2)));
            result = fzero(func, 0.3);
        end
        
        %________________________________________________________________ calculatePostShockVelocity
        function result = calculatePostShockVelocity(obj, postPressure)
            g      = obj.GAMMA;
            result = 2*sqrt(g)/(g - 1)*(1 - postPressure^((g - 1)/(2*g)));
        end
        %____________________________________________________________________ calculatePostShockMass
        function result = calculatePostShockMass(obj, postPressure)
            g      = obj.GAMMA;
            G      = (g-1)/(g+1);
            m2     = obj.mass(end);
            p2     = obj.pressure(end);
            
            result = m2*((postPressure/p2) + G) / (1 + G*(postPressure/p2));
        end
        
        %_______________________________________________________________________ calculateShockSpeed
        function result = calculateShockSpeed(obj, postMass, postVelocity)
            m2     = obj.mass(end);
            
            result = postVelocity*(postMass/m2) / ((postMass/m2) - 1);
        end
        
        %__________________________________________________________________ calculatePostContactMass
        function result = calculatePostContactMass(obj, postPressure)
            g      = obj.GAMMA;
            p1     = obj.pressure(1);
            
            result = obj.mass(1)*(postPressure/p1)^(1/g);
        end
        
        %____________________________________________________________ calculateRarefactionSoundSpeed
        function result = calculateRarefactionSoundSpeed(obj, x0, time)
            g         = obj.GAMMA;
            G         = (g-1)/(g+1);
            c1        = obj.soundSpeed(1);
            positions = obj.x;
            
            result    = G*((x0-positions)./time) + (1 - G)*c1;
        end
        
        %______________________________________________________________ calculateRarefactionVelocity
        function result = calculateRarefactionVelocity(obj, x0, time)
            g         = obj.GAMMA;
            G         = (g-1)/(g+1);
            c1        = obj.soundSpeed(1);
            positions = obj.x;
            
            result    = (1 - G)*((positions-x0)/time + c1);
        end
        
        %___________________________________________________________________ calculateRarefactionMass
        function result = calculateRarefactionMass(obj, rarefactionSoundSpeed)
            g      = obj.GAMMA;
            m1     = obj.mass(1);
            c1     = obj.soundSpeed(1);
            
            result = m1.*(rarefactionSoundSpeed./c1).^(2/(g-1));
        end
        
        %______________________________________________________________ calculateRarefactionPressure
        function result = calculateRarefactionPressure(obj, rarefactionMass)
            g      = obj.GAMMA;
            m1     = obj.mass(1);
            p1     = obj.pressure(1);
            
            result = p1*(rarefactionMass./m1).^g;
        end
        
    end%PROTECTED
    
    %==============================================================================================
    methods (Static = true) %                                                                                                          S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
