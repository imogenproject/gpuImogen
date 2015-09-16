classdef ODESolver < handle
% This class implements a large number of methods for beating a ODE to death
% It calculates the solution to the equation y' = f(y, t)
% f, below, must be a function handle and must accept vector arguments for y and t.
%___________________________________________________________________________________________________ 

%===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
        methodExplicit = 1;
        methodImplicit = 2;
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    end %PUBLIC

%==================================================================================================
    properties (SetAccess = protected, GetAccess = public)
        solution; % column vectors [x, y(x)]
        stepsize;
        integrate;
        f; % @function(x, y);
        h; % real > 0 constant step size
    end; %READONLY

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
        f_history; % Storage of previous results
        methID;
        methMatrix;% Weighting coefficients derived in Mathematica

        requireBootstrap;
        requireIC;
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        function self = ODESolver(f, x0, y0, h0, method, methType)

        self.integrate = @self.integrateNotSet;
        requireBootstrap = 1;

        if nargin >= 2; self.setODE(f); end
        if nargin >= 4; self.setInitialCondition(x0, y0); end
        if nargin >= 5; self.setStep(h0); end
        if nargin >= 7; self.setMethod(method, methType); end

        end

        % This takes the function handle 'func' which will tell us about the ODE,
        % and the parameter funcType which tells us if func returns only
        % y = f(x,y) (simply evaluates the ODE and we do our own derivaitves)
        % or y^(i) = f(x,y), the f and all derivatives requires by the method
        function status = setODE(self, func, symDerivs)
            self.f = func;
            status = 0;
        end

        % This takes a method ID (set of constraints) and stores the method matrix
        % which was computed using Mathematica
        function status = setMethod(self, methID, methType)
            methID = 1.0*(methID == 1); % convert to binary just in case
            self.methID = methID;

            switch methType
                case self.methodExplicit
                    omat = ODE_Matrix('odedb_explicit.mat');
                    self.methMatrix = omat.retreiveMethod(methID);
                    self.integrate = @self.integrateExplicit;
                case self.methodImplicit
                    omat = ODE_Matrix('odedb_implicit.mat');
                    self.methMatrix = omat.retreiveMethod(methID);
                    self.integrate = @self.integrateImplicit;
                otherwise
                    error('methType must be self.methodExplicit or self.methodImplicit');
            end
        
            self.requireBootstrap = 1;
            self.requireIC = 1;
            self.f_history = zeros(size(self.methMatrix));

        end

        function status = setStep(self, h)
            self.stepsize = h;
            self.f_history = zeros(size(self.methMatrix));
            self.f_history(:,1) = self.f(self.solution(end,1), self.solution(end,2));

            self.requireBootstrap = 1; % For now... Given we have past history we could nominally take substeps without extremely expensive implicit RK methods
        end

        function status = setInitialCondition(self, x0, y0)
            self.solution = [x0, y0];
            self.f_history(:,1) = self.f(x0,y0);

            self.requireBootstrap = 1; % Must completely re-initialize if starting over
            self.requireIC = 0;

            status = 0;
        end

        function status = integrateNotSet(self, L, abort_cond)
            error('integrate() called but setMethod() has not been. See help()');
        end

        function status = integrateImplicit(self, L, abort_cond)
            if self.requireIC == 1
                error('Cannot initiate integration without initial conditions: self.setInitialCondition(x0, y0);');
            end

            if self.requireBootstrap == 1
                self.bootstrapHistory();
            end

            nder = size(self.methMatrix,1);
            npts = size(self.methMatrix,2);

            h = self.stepsize;
            n = ceil(L/h);

            % Calculate the Step weighting Matrix for constant step size h
            sm = ones(size(self.methMatrix));
            for i = 1:nder; sm(i,:) = h^i; end

            % elementwise product it with the method weighting coefficients
            sm = sm .* self.methMatrix;

            % For the implicit method, we have y_n+1 + f(y_n+1) = stuff(y_n and previous).
            % Only y_n+1 is dynamic so y_n and before is a constant during the nonlinear iteration
	    pastMeth = sm(:,2:end);
	    nowMeth = sm(:,1)';

            for i = 1:n
                y0 = self.solution(end,2);
                f_past = self.f_history(:,1:(end-1));
                c0 = sum(pastMeth(:).*f_past(:)) + y0;
		x_next = self.solution(end,1)+h;
                
                EFF = @(x,y) y - nowMeth*self.f(x,y) - c0;
                ynext = self.NR_Solve(EFF, x_next, self.solution(end,2));

                self.solution(end+1,:) = [x_next ynext];

                % rotate function history right and insert new column
		self.f_history = [self.f(x_next, ynext) f_past];
            end
        end

        function status = integrateExplicit(self, L, abort_cond)
            if self.requireICs == 1
                error('Cannot initiate integration without initial conditions: self.setInitialCondition(x0, y0);');
            end

            if self.requireBootstrap == 1
                self.bootstrapHistory();
            end

            nder = size(self.methMatrix,1);
            npts = size(self.methMatrix,2);

            h = self.stepsize;
            n = ceil(L/h);

            % Calculate the Step weighting Matrix for constant step size h
            sm = ones(size(self.methMatrix));
            for u = 1:nder; sm(u,:) = h^u; end;
            % elementwise product it with the method weighting coefficients
            sm = sm .* self.methMatrix;

            for i = 1:n
                delta = self.evaluateMethod(sm, self.f_history);

                self.solution(end+1,:) = self.solution(end,:) + [h delta];

                % rotate function history right
                self.f_history = circshift(self.f_history, [0 1]);

                % generate new column at left
                self.f_history(:,1) = self.f(self.solution(end,1),self.solution(end,2));
            end

        end

        % Solves scalar nonlinear equation using Newton-Raphson
        function y = NR_Solve(self, f, x, y0)
            eyestep = 1i*4.9303806576313240e-32; % epsilon ^ 2
            eyemag = 4.9303806576313240e-32;

            y = y0;
            for n = 1:20
%                y_pred = y - .5*f(x, y) * (eyemag / imag(f(x, y + eyestep)));
                delta = -f(x,y) * eyemag / imag(f(x,y+eyestep));
                y = y + delta;
                if abs(delta / y) < 8*eps; break; end
            end
        end

        function y = NR_SolveAutonomous(self, f, y0)
            eyestep = 1i*4.9303806576313240e-32; % epsilon ^ 2
            eyemag = 4.9303806576313240e-32;

            y = y0;
            for n = 1:20
                delta = -f(y) * (eyemag / imag(f(y + eyestep)));
                y = y + delta;
                if abs(delta / y) < 8*eps; break; end
            end
        
        end


        % Solves vector nonlinear equation using Newton-Raphson
        function y = MV_NR_Solve(self, f, x, y0)
            eyestep = 1i*4.9303806576313240e-32; % epsilon ^ 2
            eyemag = 4.9303806576313240e-32;

            y = y0;
            D = numel(y0);
            J = zeros([D D]);
            cstep = zeros([D 1]); cstep(1) = eyestep;

            for n = 1:20
                % Evaluate J directly because we assume dimension is small
                for a = 1:D
                    J(:,a) = imag(f(x, y0 + cstep))/eyemag;
                    cstep = circshift(cstep,[1 0]);
                end

                delta = -(J^-1)*f(x, y);
                y = y + delta;
                if mean(abs(delta ./ y)) < 8*eps; break; end
            end

        end

        function y = MV_NR_SolveAutonomous(self, f, y0)
            eyestep = 1i*4.9303806576313240e-32; % epsilon ^ 2
            eyemag = 4.9303806576313240e-32;

            y = y0;
            D = numel(y0);
            J = zeros([D D]);
            cstep = zeros([D 1]); cstep(1) = eyestep;

%history = [];

            for n = 1:20
                % Evaluate J directly because we assume dimension is small
                for a = 1:D
                    J(:,a) = imag(f(y + cstep))/eyemag;
                    cstep = circshift(cstep,[1 0]);
                end

                delta = -(J^-1)*f(y);
%                ymid = y + .5*deltaA;
%		for a = 1:D
%                    J(:,a) = imag(f(ymid + cstep))/eyemag;
%                    cstep = circshift(cstep,[1 0]);
%                end
%		delta = -(J^-1)*f(y);
		y = y + delta;

%history(:,end+1) = delta;
                if mean(abs(delta ./ y)) < 8*eps; break; end
            end

%diff(log(abs(history')))

        end

        function help(self)
            disp('How to start ODESolver in a single go:');
            disp('S = ODE_Solver([@f(x, y), D, x0, y0, h0, method, methType])');
            disp('Where @f is the ode y''=f(x,y) to solve, D (if true) indicates that the function returns a vector of (not approximated) derivatives of f equal in number to that required by the method matrix, x0, y0 and h0 give initial conditions and step size, method is a HCAM/HCAB constraint matrix and methType is ODESolver.methodExplicit or ODESolver.methodImplicit');
            disp('Consider also the sequence:');
            disp('S = ODE_Solver()');
            disp('S.setODE(@f(x,y), tf)');
            disp('S.setMethod(method, methType)');
            disp('S.setInitialCondition(x0, y0)');
            disp('S.setStep(h)');
            disp('Now either way the solve can be run:');
            disp('S.integrate(length)');

        end

    end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

	function demonstrateMe(self)
	    a = 1; b = -.1;
	    d = 50;
	    u = -.5;
	    v = 4*pi;

	    y0 = 1;
	    x0 = 0;

	    F = @(x) d*exp(u*x).*(u*cos(v*x)+v*sin(v*x))/(u^2+v^2);
	    Theta = log(y0/(a+b*y0)) - a*F(0);
	    K = @(x) exp(a*F(x) + Theta);

	    exactSolution = @(x) a*K(x)./(1-b*K(x));

	    self.setODE( @(x,y) d*(a*y + b*y.^2).*exp(u*x).*cos(v*x));
	    self.setMethod([1 1 1 1], self.methodImplicit);
	    self.setInitialCondition(x0, y0);
            self.setStep(.02);

	    self.integrate(10);

%	    hold off;
	    xval = self.solution(:,1);
	    yhat = exactSolution(xval);

%	    hold on;	    

self.solution(1:5,2) - yhat(1:5)
 %           plot(xval,self.solution(:,2)-yhat,'b');

%	    self.setODE( @(x,y) [ d*(a*y + b*y.^2).*exp(u*x).*cos(v*x); d*exp(u*x).*y.*(a + b*y).*(u*cos(v*x) + d*exp(u*x).*(a + 2*b*y).*cos(v*x).^2 - v*sin(v*x)) ] );	
	    self.setODE( @(x,y) self.sampleODE(x,y));

	    self.setMethod([1 1 1 1; 1 1 1 1; 1 1 1 1], self.methodImplicit);
	    self.setInitialCondition(x0, y0);
	    self.setStep(.015);


	    self.integrate(10);
	    xval = self.solution(:,1);
            yhat = exactSolution(xval);

self.solution(1:5,2) - yhat(1:5)
% 	    plot(xval, self.solution(:,2)-yhat,'k');

	end
	
	function F = sampleODE(self, x, y)
	    a=1; b=-.1;
	    d=50;
            u=-.5; v=4*pi;

	    G    = d*exp(u*x).*cos(v*x);
	    f    = y.*(a+b*y)*G;
	    Gp   = d*exp(u*x).*(u*cos(v*x)-v*sin(v*x));
	    fp   = Gp*y*(a + b*y) + f*G*(a + 2*b*y);
	    Gpp  = d*exp(u*x)*((u^2 - v^2)*cos(v*x) - 2*u*v*sin(v*x));
            fpp  = Gpp*y*(a + b*y) + 2*f*Gp*(a + 2*b*y) + G*(2*b*f.^2 + fp*(a + 2*b*y));
	    F = [f;fp;fpp];
	end

    end%PUBLIC
        
%===================================================================================================        
    methods (Access = protected) %                                      P R O T E C T E D    [M]
        function delta = evaluateMethod(self, sm, y_history)
            delta = sum(sm(:).*y_history(:));
        end

        function q = domethod(self, M, h, xn, yn)
            q = zeros([size(M,3) 1]);
	    hf = cumprod(ones(size(M))*h,1).*M;

	    for i = 1:size(M,3)
                for j = 1:numel(xn);
                    q(i) = q(i) + sum(hf(:,j,i).*self.f(xn(j),yn(j)));
                end
	    end
        end

        % This generates history using multi-implicit methods given nothing but the starting
        % point
        function status = bootstrapHistory(self)
            maxder = size(self.methMatrix,1);
            needpoints = size(self.methMatrix,2) - 1;

            if needpoints == 0; status = 0; self.requireBootstrap = 0; return; end

            h = self.stepsize;

            % First generate a crude approximation to input to the MVNR loop
            ders = self.f(self.solution(end,1),self.solution(end,2));

            y0 = self.solution(end,2);
            ypoints = zeros(needpoints, 1);
            % Increasing index looks back in time to the past
            for a = 1:needpoints; ypoints(a) = self.solution(end,2) + ders(1)*(needpoints+1-a)*h; end

            if needpoints == 1
                xi = [h 0] + self.solution(end,1);

                switch maxder
                    case 1
                        F = @(y) y0 + self.domethod(1, h, xi(1), y) - y; % 3rd order accurate locally
                    case 2
                        F = @(y) y0 + self.domethod( [.5 .5; -1/12 1/12], h, xi, [y y0]) - y; % 5th order accurate locally
                    case 3
                        F = @(y) y0 + self.domethod( [.5, .5; -1/10, 1/10; 1/120, 1/120], h, xi, [y y0]) - y; % 7th order accurate locally
                end
            elseif needpoints == 2
                xi = h*[2 1 0] + self.solution(end,1);
                %[most recent; middle; y0]
                switch maxder
                    case 1
			coeffMatrix = reshape([5 8 -1 -1 8 5]/12,[1 3 2]);
                    case 2
			coeffMatrix = reshape([101 -13 128 40 11 3 11 -3 128 -40 101 13]/240,[2 3 2]);
                    case 3
			coeffMatrix = reshape([17007 -2727 169 24576 5040 1024 -1263 -423 -41 -1263 423 -41 24576 -5040 1024 17007 2727 169 ]/40320,[3 3 2]);
                end
                F = @(y) [y(2)-y(1); y0-y(2)] + self.domethod(coeffMatrix, h, xi, [y; y0]); % 10th order
            elseif needpoints == 3
		xi = h*[3 2 1 0] + self.solution(end,1);
		switch maxder
		    case 1 % 5th order
			coeffMatrix = reshape([9 19 -5 1 -1 13 13 -1 1 -5 19 9]/24,[1 4 3]);
		    case 2 % 10th order
			coeffMatrix = reshape([34465 -3849 42255 22977 12015 7263 1985 489 1215 -279 44145 -9153 44145 9153 1215 279 1985 -489 12015 -7263 42255 -22977 34465 3849]/90720, [2 4 3]);
		    case 3 % 15th order
			coeffMatrix = reshape([4562615 -644829 34107 9605385 863217 597105 -2369655 -863217 -201231 176695 53469 4539 -62775 18141 -1467 6050295 -1194993 135567 6050295 1194993 135567 -62775 -18141 -1467 176695 -53469 4539 -2369655 863217 -201231 9605385 -863217 597105 4562615 644829 34107]/11975040, [3 4 3]);
		end
		F = @(y) [y(2)-y(1); y(3)-y(2); y0-y(3)] + self.domethod(coeffMatrix, h, xi, [y; y0]);
	    end
            % Perform nonlinear iterations to solve the system for multiple future values of y simultaneously
            ysol = self.MV_NR_SolveAutonomous(F, ypoints);

            self.f_history(:,1) = self.f(self.solution(end,1), y0);

            for n = 1:needpoints
                self.solution(end+1,:) = [self.solution(end,1)+h ysol(needpoints+1-n)];
            end
            
            theta = self.solution((end-needpoints):end,:);
            for n = 1:(needpoints+1)
                self.f_history(:,needpoints+2-n) = self.f(theta(n,1), theta(n,2));
            end

        end

    end%PROTECTED
                
%===================================================================================================        
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
        
end%CLASS
