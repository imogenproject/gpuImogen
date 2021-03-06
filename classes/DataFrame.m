classdef DataFrame < handle
    % Class contains a loaded Imogen dataframe
    % Provides a flat interface to variables and derived cell state functions independent of the
    % actual format of storage (primitive or conservative)
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, GetAccess = protected) %                 C O N S T A N T         [P]
        fmtConservative = 1;
        fmtPrimitive = 2;
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                       P U B L I C  [P]
        time;
        parallel;
        gamma;
        about;
        ver;
        dGrid;
        magX, magY, magZ;
    end %PUBLIC
    
    properties (Dependent = true, SetAccess = private) %                    D E P E N D E N T [P]
        % These are the primitive and conservative forms of all variable fields generated by Imogen
        % Depending on the form they are stored in,
        mass;
        momX, momY, momZ, mom;
        velX, velY, velZ, vel;
        ener, eint, pressure;
        
        mass2;
        momX2, momY2, momZ2, mom2;
        velX2, velY2, velZ2, vel2;
        ener2, eint2;
        
        % Derived properties
        speed, speed2;
        soundspeed, soundspeedL, soundspeedH;
        temperature;
        % if two fluids, 'soundspeed' picks L
        % L -> low K, perfectly coupled lower soundspeed
        % H -> high K, dust-uncoupled higher ('normal') soundspeed
        
        deltavX, deltavY, deltavZ, deltav;
        % two fluid differential speeds
        
    end %DEPENDENT
    
    %===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %            P R O T E C T E D [P]
        pInternalVarFmt;
        pTwoFluids;
        % SOME - I repeat SOME - of these are the actual fields that were loaded
        % which ones can be safely accessed depends on the state of varFormat
        pMass;
        pMomX, pMomY, pMomZ;
        pVelX, pVelY, pVelZ;
        pEner, pEint;
        
        pMass2;
        pMomX2, pMomY2, pMomZ2;
        pVelX2, pVelY2, pVelZ2;
        pEner2, pEint2;
        
        pTrueShape;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                 G E T / S E T  [M]
        function self = DataFrame(input)
           if isa(input, 'struct')
               frame = input;
           elseif isa(input, 'char')
               %frame = util_LOadWHoleFrame(
           end
           
           self.pMass = frame.mass;
           if isfield(frame, 'momX'); self.pInternalVarFmt = self.fmtConservative; else; self.pInternalVarFmt = self.fmtPrimitive; end
           if isfield(frame, 'mass2'); self.pTwoFluids = 1; else; self.pTwoFluids = 0; end
           
           self.pPopulateFluidFields(input);
           self.pPopulateAttributeFields(input);
    
           self.pTrueShape = size(self.pMass);
           
           self.magX = 0; self.magY = 0; self.magZ = 0;
        end
        
        function needtoresave = checkpinteg(self)
            if prod(self.pTrueShape) ~= numel(self.mass)
                fprintf('WARNING: self.pTrueShape WRONG SIZE: SAVE GLITCH CAUGHT: AUTOFIXING\n');
                self.pTrueShape = [size(self.mass,1) 1 1 size(self.mass,2)];
                if prod(self.pTrueShape) ~= numel(self.mass)
                    self.pTrueShape(4) = size(self.mass,4);
                end
                self.unsquashme;
                needtoresave = 1;
            else
                needtoresave = 0;
            end
        end

        % basic rho/v/p/E for fluid 1
        function y = get.mass(self); y = self.pMass; end 
        function y = get.momX(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomX; else; y = self.pMass.*self.pVelX; end; end
        function y = get.velX(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomX./self.mass; else; y = self.pVelX; end; end
        function y = get.momY(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomY; else; y = self.pMass.*self.pVelY; end; end
        function y = get.velY(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomY./self.mass; else; y = self.pVelY; end; end
        function y = get.momZ(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomZ; else; y = self.pMass.*self.pVelZ; end; end
        function y = get.velZ(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomZ./self.mass; else; y = self.pVelZ; end; end

        function y = get.vel(self); y.X = self.velX; y.Y = self.velY; y.Z = self.velZ; end
        function y = get.vel2(self); y.X = self.velX2; y.Y = self.velY2; y.Z = self.velZ2; end
        function y = get.mom(self); y.X = self.momX; y.Y = self.momY; y.Z = self.momZ; end
        function y = get.mom2(self); y.X = self.momX2; y.Y = self.momY2; y.Z = self.momZ2; end
        
        function y = get.ener(self); if self.pInternalVarFmt == self.fmtConservative %#ok<ALIGN>
                y = self.pEner;
            else
                y = self.pEint + .5*self.pMass .* (self.pVelX.^2+self.pVelY.^2+self.pVelZ.^2);
            end
        end
        function y = get.eint(self); if self.pInternalVarFmt == self.fmtConservative %#ok<ALIGN>
                y = self.pEner - .5*(self.pMomX.^2+self.pMomY.^2+self.pMomZ.^2)./self.pMass;
            else
                y = self.pEint;
            end
        end
        function y = get.pressure(self); y = (self.gamma-1)*self.eint; end
        
        function y = get.soundspeed(self)
            % The normal (gas) sound speed; If two fluids, alias for the slower (coupled) c_s
            y = self.soundspeedL;
        end
            
        function y = get.soundspeedL(self)
            gg1 = self.gamma*(self.gamma-1);
            if self.pTwoFluids
                y = sqrt(gg1*self.eint ./ (self.pMass+self.pMass2));
            else
                y = sqrt(gg1*self.eint ./ self.pMass);
            end
        end
                
        function y = get.soundspeedH(self)
            y = sqrt(self.gamma*(self.gamma-1)*self.eint ./ self.pMass);
        end
        
        % basic rho/v/p/E for fluid 2
        function y = get.mass2(self); y = self.pMass2; end 
        function y = get.momX2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomX2; else; y = self.pMass2.*self.pVelX2; end; end
        function y = get.velX2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomX2./self.mass2; else; y = self.pVelX2; end; end
        function y = get.momY2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomY2; else; y = self.pMass2.*self.pVelY2; end; end
        function y = get.velY2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomY2./self.mass2; else; y = self.pVelY2; end; end
        function y = get.momZ2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomZ2; else; y = self.pMass2.*self.pVelZ2; end; end
        function y = get.velZ2(self); if self.pInternalVarFmt == self.fmtConservative; y = self.pMomZ2./self.mass2; else; y = self.pVelZ2; end; end
        function y = get.ener2(self); if self.pInternalVarFmt == self.fmtConservative %#ok<ALIGN>
                y = self.pEner2;
            else
                y = self.pEint2 + .5*self.pMass2 .* (self.pVelX2.^2+self.pVelY2.^2+self.pVelZ2.^2);
            end
        end

        function y = get.speed(self); if self.pInternalVarFmt == self.fmtConservative %#ok<ALIGN>
                y = sqrt(self.pMomX.^2+self.pMomY.^2+self.pMomZ.^2)./self.mass;
            else
                y = sqrt(self.pVelX.^2+self.pVelY.^2+self.pVelZ.^2);
            end
        end
        function y = get.speed2(self); if self.pInternalVarFmt == self.fmtConservative %#ok<ALIGN>
                y = sqrt(self.pMomX2.^2+self.pMomY2.^2+self.pMomZ2.^2)./self.mass2;
            else
                y = sqrt(self.pVelX2.^2+self.pVelY2.^2+self.pVelZ2.^2);
            end
        end
        
        function y = get.temperature(self); y = (self.gamma-1)*self.eint ./ self.mass; end
        
        function y = get.deltavX(self)
            if self.pTwoFluids == 0; y = []; return; end
            if self.pInternalVarFmt == self.fmtConservative
                y = self.pMomX2./self.pMass2 - self.pMomX ./ self.pMass; else; y = self.pVelX2 - self.pVelX; end
        end
        function y = get.deltavY(self)
            if self.pTwoFluids == 0; y = []; return; end
            if self.pInternalVarFmt == self.fmtConservative
                y = self.pMomY2./self.pMass2 - self.pMomY ./ self.pMass; else; y = self.pVelY2 - self.pVelY; end
        end
        function y = get.deltavZ(self)
            if self.pTwoFluids == 0; y = []; return; end
            if self.pInternalVarFmt == self.fmtConservative
                y = self.pMomZ2./self.pMass2 - self.pMomZ ./ self.pMass; else; y = self.pVelZ2 - self.pVelZ; end
        end
        function y = get.deltav(self); y.X = self.deltavX; y.Y = self.deltavY; y.Z = self.deltavZ; end

    end%GET/SET

    %===================================================================================================
    methods (Access = public) %                                                 P U B L I C  [M]
        
        function f = toSaveframe(self)
           % DataFrame.toSaveframe() processes its internal state and emits the save frame structure it came from.
           % this process is lossless.
           
           f = struct();
           fields = {'time','parallel','gamma','about','ver','dGrid'};
           
           for q = 1:numel(fields)
               f.(fields{q}) = self.(fields{q});
           end
           
           f.mass = self.mass;
           f.momX = self.momX;
           f.momY = self.momY;
           f.momZ = self.momZ;
           f.ener = self.ener;
           
           if self.pTwoFluids
               f.mass2 = self.mass2;
               f.momX2 = self.momX2;
               f.momY2 = self.momY2;
               f.momZ2 = self.momZ2;
               f.ener2 = self.ener2;
           end
           
           f.magX = 0;
           f.magY = 0;
           f.magZ = 0;
        end
        
        function concatFrame(self, F)
            fieldsA = {'pMass','pMomX','pMomY','pMomZ','pEner'};
            fieldsB = {'mass','momX','momY','momZ','ener'};
            
            for q = 1:5
                self.(fieldsA{q}) = cat(4, self.(fieldsB{q}), F.(fieldsB{q}));
            end
            
            if isfield(self.time, 'time') && isfield(F.time, 'time')
                self.time.time = [self.time.time; F.time.time]; 
            end
            self.time.iterMax = F.time.iterMax;
            self.time.iteration = F.time.iteration;
            
            self.pTrueShape = size(self.mass);
        end
        
        function truncate(self, x, y, z, t)
            if (nargin < 5) || isempty(t)
                t = 1:size(self.mass,4);
            end
            if (nargin < 4) || isempty(z)
                z = 1:size(self.mass,3);
            end
            if (nargin < 3) || isempty(y)
                y = 1:size(self.mass, 2);
            end
            if (nargin < 2) || isempty(x)
                x = 1:size(self.mass, 1);
            end
            
            f2 = self.pNamesOfInternalFields();
            
            for j = 1:numel(f2)
                b = self.(f2{j});
                self.(f2{j}) = b(x, y, z, t);
            end
            
            if numel(size(self.mass)) ~= self.pTrueShape % if we've been squished
                lst = {x,y,z,t};
                self.time.time = self.time.time(lst{numel(size(self.mass))});
                self.pTrueShape = [numel(x) numel(y) numel(z) numel(t)];
            else
                self.time.time = self.time.time(t);
                self.pTrueShape = size(self.mass);
            end
            
            
        end
        
        function patchBadTimeslice(self, tIndex)
            disp('WARNING: This function patches a bad time slice by averaging adjacent slices.')
            disp('WARNING: This is not valid for transient structures!');
            disp('WARNING: This should be used ONLY in the event that a failed resume leaves behind a partly-written garbage frame');
            
            F = self.pNamesOfInternalFields();
            
            for j = 1:numel(F)
                b = self.(F{j});
                b(:,:,:,tIndex) = .5*(b(:,:,:,tIndex-1) + b(:,:,:,tIndex+1));
                self.(F{j}) = b;
            end

        end
        
        function n = chopOutAnomalousTimestep(self)
            % .chopOutAnomalousTimestep()
            % Resumed runs save a frame immediately after taking one step, which it not usually what we want.
            % This looks for frames separated by only a single timestep and chops them out.
            % for sanity reasons, it limits itself to cutting out a maximum of 10 frames

            tau = diff(self.time.time);
            t0 = mean(tau(round(end/2):end));
    
            % assuming at least 50 timesteps per saveframe - safe
            b = find((tau < t0/50) & (tau > 0))';
    
            if numel(b) > 10
                fprintf('F.chopOutAnomalousTimestep: Apparently found %i anomalous frames... doing nothing, I am probably wrong.\n', numel(b));
            end

            if numel(b) > 0
                fprintf('| F.chopOutAnomalousTimestep: deleting\n');
            end
            
            for x = 1:numel(b)
                fprintf('frame %i has dt=%f < .5 t0=%f\n', int32(b(x)+1), tau(b(x)), t0);
            end
            
            b = [0, b+1, numel(self.time.time)];
            
            rng = [];
            for x = 1:(numel(b)-1)
                rng = [rng (b(x)+1):(b(x+1)-1)];
            end
            rng = [rng b(end)];
            
            self.truncate([], [], [], rng);
    
            n = numel(b)-2;
        end
        
        function tf = checkForBadRestartFrame(self, doit)
           % This function looks for a huge negative jump in diff(self.time.time)
           % If this has occurred, some bulls*** has caused a run to restart in the middle of
           % itself instead of at the end: Chops out the large range of frames in the middle.
           
           ohno = find(diff(self.time.time) < 0);
           ohno = ohno + 1; % this picks the first stupidly placed restart frame
           tf = 0;
           
           if numel(ohno) > 0
               N = size(self.time.time,4);
               
               fprintf('| negative time jump at frame %i ', int32(ohno) );
               e0 = self.time.time(ohno);
               tau = self.time.time(ohno+1)-self.time.time(ohno);
               
               for j = 1:(ohno-1)
                   if self.time.time(j) > e0 - 3*tau/2; break; end
               end
               
               if nargin < 2
                   fprintf('Proposed juncture times:\n');
                   disp(self.time.time([(j-4):j ohno:(ohno+4)])');
                   
                   dtproper = max(abs(diff(self.time.time([(j-4):j ohno:(ohno+4)]), 2) ));
                   if any(dtproper > 1e-4)
                       fprintf('Expected frame juncture: [1:%i %i:end]\n',j, ohno);
                       fprintf('Times: ');
                       disp(self.time.time([(j-4):j ohno:(ohno+3)]));
                       warning('Found max delta^2 T/delta frame^2 > 1e-4 -> unacceptable, will not truncate\n');
                       doit = 0;
                   else
                       doit = 1;
                   end
                   
               end
                  
               if doit
                   self.truncate([], [], [], [1:(j-1) (ohno+2):N]);
                   plot(diff(self.time.time));
                   disp('Good to save? ');
                   tf = input('?');
                   if tf
                       tf = 1;
                   else
                       tf = 0;
                   end
               else
                   tf = 0;
               end
               
           end
            
        end
        
        function squashme(self)
           
            self.pTrueShape = size(self.mass);
            
            names = self.pNamesOfInternalFields();
            for n = 1:numel(names)
                self.(names{n}) = squeeze(self.(names{n}));
            end
            
        end
        
        function unsquashme(self)
            
            names = self.pNamesOfInternalFields();
            for n = 1:numel(names)
                self.(names{n}) = reshape(self.(names{n}), self.pTrueShape);
            end
                
        
        end
        
        function reshape(self, shape)
            if prod(shape) ~= numel(self.mass); error('To reshape, numel(x) must not change.'); end
            
            self.pTrueShape = shape;
            self.unsquashme;            
        end
        
        function staticshift(self, amt)
            u=1:size(self.mass,1);
            v=1:size(self.mass,2);
            w=1:size(self.mass,3);
            
            u = circshift(u,amt(1));
            v = circshift(v,amt(2));
            w = circshift(w,amt(3));
            
            if numel(u) > 8
                u(1:4) = 1:4;
                n = numel(u);
                u((n-3):n) = (n-3):n;
            end
            if numel(v) > 8
               v(1:4) = 1:4;
               n = numel(v);
               v((n-3):n) = (n-3):n;
            end
            if numel(w) > 8
               w(1:4) = 1:4;
               n = numel(w);
               w((n-3):n) = (n-3):n;
            end
            
            names = self.pNamesOfInternalFields();
            for n = 1:numel(names)
                q = self.(names{n});
                self.(names{n}) = q(u,v,w,:);
            end
            
        end

        function constshift(self, amt)
            error('fixme: this is not implemented yet, it is just a verbatim ^c^v of staticshift');
            u=1:size(self.mass,1);
            v=1:size(self.mass,2);
            w=1:size(self.mass,3);
            
            u = circshift(u,amt(1));
            v = circshift(v,amt(2));
            w = circshift(w,amt(3));
            
            if numel(u) > 8
                u(1:4) = 1:4;
                n = numel(u);
                u((n-3):n) = (n-3):n;
            end
            if numel(v) > 8
               v(1:4) = 1:4;
               n = numel(v);
               v((n-3):n) = (n-3):n;
            end
            if numel(w) > 8
               w(1:4) = 1:4;
               n = numel(w);
               w((n-3):n) = (n-3):n;
            end
            
            names = self.pNamesOfInternalFields();
            for n = 1:numel(names)
                q = self.(names{n});
                self.(names{n}) = q(u,v,w,:);
            end
            
        end
        
        function circshift(self, amt)
            if numel(amt) == numel(size(self.mass))
                names = self.pNamesOfInternalFields();
                for n = 1:numel(names)
                    self.(names{n}) = circshift(self.(names{n}), amt);
                end
            else
                error('asdf');
            end
        end
        
        function remap(self, m)
            if numel(m) == size(self.mass,1)
                names = self.pNamesOfInternalFields();
                for n = 1:numel(names)
                    x = self.(names{n});
                    
                    self.(names{n}) = x(m,:,:,:);
                end
            else
                error('asdf');
            end
        end
        
        function O = vorticity(self, direct, BC)
            % This function (not a get/set) computes the vorticity
            % direct may be 1/2/3=X/Y/Z, 0 = |curl(\vec{v})|
            % bc is drawn from ENUM.BCMODE_*; Implemented options:
            % BCMODE_CIRCULAR, BCMODE_STATIC
            % bc must be a structure of the type enumerated by the
            % initializer function.
            
            BC = BCManager.expandBCStruct(BC);
            if direct == 1
                m = self.velZ;
                O = (circshift(m, [0 -1 0])-circshift(m, [0 1 0]))/(2*self.dGrid{2}); % (Vz)_y
                m = self.velY;
                O = O - (circshift(m, [0 0 -1])-circshift(m, [0 0 1]))/(2*self.dGrid{3}); %-(Vy)_z
            elseif direct == 2
                m = self.velX;
                O = (circshift(m, [0 0 -1])-circshift(m, [0 0 1]))/(2*self.dGrid{3}); % +(Vx)_z
                m = self.velZ;
                O = O-(circshift(m, [-1 0 0])-circshift(m, [1 0 0]))/(2*self.dGrid{1}); % -(Vz)_x
            elseif direct == 3
                m = self.velY;
                O = (circshift(m, [0 -1 0])-circshift(m, [0 1 0]))/(2*self.dGrid{1}); % (Vy)_x
                m = self.velX;
                O = O -(circshift(m, [-1 0 0])-circshift(m, [1 0 0]))/(2*self.dGrid{1}); % -(Vx)_y
                
            elseif direct == 0
                O = self.vorticity(1, BC).^2;
                O = O + self.vorticity(2,BC).^2;
                O = O + self.vorticity(3,BC).^2;
                O = sqrt(O);
            end
        end
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function nof = pNamesOfInternalFields(self)
            if self.pInternalVarFmt == self.fmtConservative
                if self.pTwoFluids
                    % conservative format two fluids
                    nof = {'pMass', 'pMomX', 'pMomY', 'pMomZ', 'pEner', 'pMass2', 'pMomX2', 'pMomY2', 'pMomZ2', 'pEner2'};
                else
                    % conservative format one fluid
                    nof = {'pMass', 'pMomX', 'pMomY', 'pMomZ', 'pEner'};
                end
            else
                if self.pTwoFluids
                    % conservative format two fluid
                    nof = {'pMass', 'pVelX', 'pVelY', 'pVelZ', 'pEint', 'pMass2', 'pVelX2', 'pVelY2', 'pVelZ2', 'pEint2'};
                else
                    nof = {'pMass', 'pVelX', 'pVelY', 'pVelZ', 'pEint'};
                end
            end
        end
        
        function pPopulateFluidFields(self, frame)
            if self.pInternalVarFmt == self.fmtConservative
               if self.pTwoFluids
                   % conservative format two fluids
                   f1 = {'mass',  'momX',  'momY',  'momZ',  'ener',  'mass2',  'momX2',  'momY2',  'momZ2',  'ener2'};
                   f2 = {'pMass', 'pMomX', 'pMomY', 'pMomZ', 'pEner', 'pMass2', 'pMomX2', 'pMomY2', 'pMomZ2', 'pEner2'};
               else
                   % conservative format one fluid
                   f1 = {'mass',  'momX',  'momY',  'momZ',  'ener'};
                   f2 = {'pMass', 'pMomX', 'pMomY', 'pMomZ', 'pEner'};
               end
           else
               if self.pTwoFluids
                   % conservative format two fluid
                   f1 = {'mass',  'velX',  'velY',  'velZ',  'eint',  'mass2',  'velX2',  'velY2',  'velZ2',  'eint2'};
                   f2 = {'pMass', 'pVelX', 'pVelY', 'pVelZ', 'pEint', 'pMass2', 'pVelX2', 'pVelY2', 'pVelZ2', 'pEint2'};
               else
                   f1 = {'mass',  'velX',  'velY',  'velZ',  'eint'};
                   f2 = {'pMass', 'pVelX', 'pVelY', 'pVelZ', 'pEint'};
               end
           end
           
           for j = 1:numel(f1)
               self.(f2{j}) = frame.(f1{j});
           end
        end
        
        function pPopulateAttributeFields(self, frame)
            self.time  = frame.time;
            self.parallel = frame.parallel;
            self.gamma = frame.gamma;
            self.dGrid = frame.dGrid;
        end
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
