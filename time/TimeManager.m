classdef TimeManager < handle
    % The manager class responsible for handling time related variables and operations.
    %===============================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===============================================================================================
    properties (SetAccess = public, GetAccess = public, Transient = true) %         P U B L I C  [P]
        time;        % Simulation time.                                              double
        dTime;       % Current timestep.                                             double
        iteration;   % Current iteration.                                            int
        history;     % History of timestep values.                                   double(N)
        CFL;         % Courant-Freidrichs-Levy condition prefactor.                  double
        ITERMAX;     % Maximum iterations before finishing run.                      int
        TIMEMAX;     % Maximum simulation time before finishing run.                 double
        WALLMAX;     % Maximum wall time before finishing run in hours.              double
        wallTime;    % Number of hours since run was started.                        double
        startTime;   % Time the run was started.                                     Date Vector
        startSecs;   % UNIX timestamp at start (used to find avg iter/sec)           unsigned long
        timePercent; % Percent complete based on simulation time.                    double
        iterPercent; % Percent complete based on iterations of maximum.              double
        wallPercent; % Percent complete based on wall time.                          double
        
        lastOutputPercent; % Last [time iter wall] %age we printed status info at    double 3x1
        running;     % Specifies if the simulation should continue running.          logical
        dtAverage;   % The accumulated mean timestep                                 double

        stepsPerChkpt; % Number of steps between in-memory checkpoint dumps
        stepsSinceIncident;

        firstWallclockValue;
    end%PUBLIC
    
    %==============================================================================================
    properties (SetAccess = public, GetAccess = private) %                        P R I V A T E [P]
        parent;      % Parent manager                                                ImogenManager
    end %PRIVATE
    
    properties (SetAccess = private, GetAccess = private)
        pItersDigits; 
        pLogString;
        
        pKahanTimeC;
    end
    
    %==============================================================================================
    methods %                                                                    G E T / S E T  [M]
        
        function result = get.running(obj)
            result = (obj.iteration < obj.ITERMAX) && (obj.time < obj.TIMEMAX) ...
                && (obj.wallTime < obj.WALLMAX);
        % Note that < is correct for iterations
        % This is hit a the start of while(running) {...},
        % while obj.iteration is updated at the end of the ...,
        % thus we will see iteration = ITERMAX when we've taken the final step.
        end
        
    end%GET/SET
    
    %==============================================================================================
    methods (Access = public) %                                                    P U B L I C  [M]
        
        %______________________________________________________________________________ TimeManager
        % Creates a new TimeManager instance.
        function obj = TimeManager()
            obj.startTime   = clock;
            obj.startSecs   = tic; % totally invalid input but prevents a crash
            obj.dTime       = 0;
            
            obj.time        = 0;
            obj.wallTime    = 0;
            obj.iteration   = 0;
            obj.ITERMAX     = 1000;
            obj.TIMEMAX     = 5000;
            obj.WALLMAX     = 1e5;
            obj.history     = zeros(obj.ITERMAX,1);
            obj.iterPercent = 0;
            obj.timePercent = 0;
            obj.wallPercent = 0;
            obj.lastOutputPercent = [-100 -100 -100];
            obj.dtAverage   = 0;
            obj.pLogString  = '';
            
            obj.stepsPerChkpt=25;
            obj.stepsSinceIncident = inf;
            
            obj.pKahanTimeC = 0;
        end
        
        function nProblems = parseIni(self, ini)
            % Processes TimeManager-related settings in the initializer struct
            
            nProblems = [0 0];
            
            if isfield(ini, 'cfl')
                self.CFL = ini.cfl;
            else
                warning(['Danger: No CFL prefactor was provided; Default value of ' mat2str(self.time.CFL) ' will be used.\n']); 
                nProblems(2) = nProblems(2) + 1;
            end
            
            if isfield(ini, 'iterMax')
                self.ITERMAX = ini.iterMax;
            end
            if isfield(ini, 'timeMax')
                self.TIMEMAX = ini.timeMax;
            end
            if isfield(ini, 'wallMax')
                self.WALLMAX = ini.wallMax;
            end
            
        end
        
        function recordWallclock(obj)
            obj.firstWallclockValue = clock;
        end
        
        %___________________________________________________________________________________ update
        % Calculate the correct timestep for the upcoming flux iteration (both the forward and
        % backward steps) of the main loop according to the Courant-Freidrichs-Lewy condition
        function update(obj, fluids, mag)
            %>< run             Imogen run manager                             ImogenManager
            %>< fluids          FluidManager array
            %>< mag             magnetic field arrays                          MagnetArray(3)
            %--- Initialization ---%
            geo = obj.parent.geometry;
            
            dtMin = 1e38;

            for f = 1:numel(fluids)
                if fluids(f).checkCFL == 0; continue; end
                
                mass = fluids(f).mass;
                ener = fluids(f).ener;
                mom  = fluids(f).mom;
                
                %--- Find max velocity ---%
                %           Find the maximum fluid velocity in the grid and its vector direction.
                if obj.parent.pureHydro == 1
                    soundSpeed = cudaSoundspeed(mass, ener, mom(1), mom(2), mom(3), fluids(f).gamma);
                else
                    soundSpeed = cudaSoundspeed(mass, ener, mom(1), mom(2), mom(3), ...
                        mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, fluids(f).gamma);
                end
                
                %[cmax, gridIndex] = directionalMaxFinder(mass, soundSpeed, mom(1), mom(2), mom(3));
                currentTau = cflTimestep(fluids(f), soundSpeed, geo, obj.parent.cfdMethod);
                GPU_free(soundSpeed);
                
                unhappy = 0;
                if ~isreal(currentTau) || isnan(currentTau) || isinf(currentTau)
                    % FIXME my gpu routines will return NaN not complex
                    % FIXME and fascinatingly, NaN is apparently a real number in matlab's world
                    r = mpi_myrank();
                    fprintf('RANK %i: Simulation crashing, this rank''s computed timestep is nonreal.\n', r);
                    unhappy = 1;
                end
                mpi_errortest(unhappy);
                
                dtMin = min(dtMin, currentTau);
            end
            
            dtMin = mpi_min(dtMin);
            
            %--- Calculate new timestep ---%
            %           Using Courant-Freidrichs-Levy (CFL) condition determine safe step size
            %           accounting for maximum simulation time.
            obj.dTime = obj.CFL*dtMin;
            
            % I've used this in "unreliable" settings to (slowly) recover CFL is something
            % makes the checkpoint restorer set a ridiculously small coefficient;
            % usually that situation just means the simulation is hopeless though
            %if obj.CFL < 0.5
            %    obj.CFL = obj.CFL + .001 * (.5 - obj.CFL);
            %end
            
            newTime   = obj.time + 2*obj.dTime; % Each individual fwd or bkwd sweep is a full step in time
            if (newTime > obj.TIMEMAX)
                obj.dTime = .5*(obj.TIMEMAX - obj.time);
                %newTime = obj.TIMEMAX;
            end
            
        end
        
        %__________________________________________________________________________________ updateUI
        function updateUI(obj)
            save = obj.parent.save;
            
            %--- Clock first loops ---%
            %           Clocks the first loop of execution and uses that to determine an estimated
            %           time to complete that is displayed in the UI and log files.
            if obj.iteration < 12
                switch obj.iteration
                    
                    case 0        %Activate clock timer for the first loop
                        obj.startSecs = tic;
                        obj.pItersDigits = ceil(log10(obj.ITERMAX)); %
                        obj.pLogString = '[[ %0ni/%0ni | %3.5g/%3.5g | avg %.3g iter/s | by %s at %s ]]\n';
                        obj.pLogString(6) = sprintf('%i',obj.pItersDigits);
                        obj.pLogString(11) = sprintf('%i',obj.pItersDigits);
                    case 10        %Stop clock timer and use the elapsed time to predict total run time
                        tPerStep = toc(obj.startSecs)/10;
                        save.logPrint('\tFirst 10 timesteps averaged %0.4g secs ea.\n', tPerStep);
                        
                        if (obj.iterPercent > obj.timePercent)
                            secsRemaining = tPerStep*(obj.ITERMAX-10);
                        else
                            secsRemaining = tPerStep*ceil(obj.TIMEMAX/obj.time);
                        end
                        
                        finTime     = now + secsRemaining/86400;
                        expComplete = datestr( finTime , 'HH:MM:SS PM');
                        
                        if ( floor(finTime) - floor(now) >= 1.0 )
                            expComplete = [expComplete ' on ' datestr( finTime, 'mmm-dd')];
                        end
                        
                        dDays       = floor(secsRemaining/86400); rem = secsRemaining - 86400*dDays;
                        dHours      = floor( rem / 3600 );        rem = rem - 3600*dHours;
                        dMinutes    = floor( rem / 60 );          rem = rem - 60*dMinutes;
                        dSecs       = ceil(rem);
                        save.logPrint('\tEstimated compute time           : [%g days | %g hr | %g mins | %g sec ]\n', ...
                            dDays, dHours, dMinutes, dSecs);
                        save.logPrint('\tProjected wallclock at completion: %s\n', expComplete);
                end
            end
            
            %--- Update UI for critical loops ---%
            %           Displays information to the UI for critical points during the run.
            movement = [obj.timePercent, obj.iterPercent, obj.wallPercent] - obj.lastOutputPercent;
            doUpdate = any(movement > 10);
            
            if (doUpdate)
                [compPer, index] = max([obj.timePercent, obj.iterPercent, obj.wallPercent]);
                switch index
                    case 1
                        info = {'Time', obj.time, obj.TIMEMAX};
                    case 2
                        info = {'Iteration', obj.iteration, obj.ITERMAX};
                    case 3
                        info = {'Wall time', obj.wallTime, obj.WALLMAX};
                end
                
                infoStr = sprintf('%s: %0.5g of %0.5g', info{1}, info{2}, info{3});
                
                %--- Prepare and display the UI update string ---%
                cTime   = now;
                curTime = strcat(datestr(cTime , 'HH:MM:SS PM'),' on ', datestr(cTime, ' mm-dd-yy'));
               
                save.logPrint(obj.pLogString, obj.iteration, obj.ITERMAX, obj.time, obj.TIMEMAX, obj.iteration/toc(obj.startSecs), info{1}, curTime);
                %save.logPrint('[[ %0.3g%% | %s |  %s | avg %i iter/s ]]\n', compPer, infoStr, curTime, obj.iteration/toc(obj.startSecs));
                
                obj.lastOutputPercent = [obj.timePercent, obj.iterPercent, obj.wallPercent];
            end
            
            obj.parent.abortCheck();
            
        end
        
        %____________________________________________________________________________ updateWallTime
        function updateWallTime(obj)
            obj.wallTime    = etime(clock(), obj.startTime)/3600;
            obj.wallPercent = 100*obj.wallTime/obj.WALLMAX;
        end
        
        %______________________________________________________________________________________ step
        % Increments the iteration variable by one for the next loop.
        function step(obj)
            obj.iteration   = obj.iteration + 1;
            
            % Accumulate dt in a Kahan-compensated manner
            y = (2*obj.dTime) - obj.pKahanTimeC;
            t = obj.time + y;
            obj.pKahanTimeC = (t-obj.time)-y;
            obj.time = t;
            
            
            obj.wallTime    = etime(clock(), obj.startTime)/3600;
            
            obj.timePercent = 100*obj.time/obj.TIMEMAX;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
            obj.wallPercent = 100*obj.wallTime/obj.WALLMAX;
            
            obj.updateUI();
            obj.appendHistory();
        end
        
        function fakeStep(obj)
            % Updates the vars that determine if we will save after this step
            % Needed because we need to 'peek' at this result before stepping,
            % in order to determine whether to take a halfstep on the source operators in order
            % to have 2nd order time accuracy when saving
            obj.iteration   = obj.iteration + 1;
            obj.time        = obj.time + 2*obj.dTime;
            obj.timePercent = 100*obj.time/obj.TIMEMAX;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
        end
        function fakeBackstep(obj)
            % Reverses the result of a fakeStep
            obj.iteration   = obj.iteration - 1;
            obj.time        = obj.time - 2*obj.dTime;
            obj.timePercent = 100*obj.time/obj.TIMEMAX;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
        end
        
        %__________________________________________________________________________________ toStruct
        % Converts the TimeManager object to a structure for saving and non-class use.
        % # result        The structure resulting from conversion of the TimeManager object.  Struct
        function result = toStruct(obj)
            result.time       = obj.time;
            result.history    = obj.history(1:obj.iteration);
            result.iterMax    = obj.ITERMAX;
            result.timeMax    = obj.TIMEMAX;
            result.wallMax    = obj.WALLMAX;
            result.started    = datestr( obj.startTime );
            result.iteration  = obj.iteration;
        end
        
        % Converts the existing time struct into a nice shiny
        function resumeFromSavedTime(obj, elapsed, newlimit)
            % Fields in new limit value:
            %newlimit.itermax = 20;
            %newlimit.timemax = 100;
            %newlimit.frame = 8;
            
            obj.time       = elapsed.time;
            if isfield(newlimit, 'itermax')
                obj.parent.save.PERSLICE = obj.parent.save.PERSLICE * obj.ITERMAX / newlimit.itermax;
                obj.parent.save.logPrint('    Resume notice: Automatically rescaling all save rates by %f.\n', newlimit.itermax/obj.ITERMAX);
                obj.ITERMAX    = newlimit.itermax;
            end
            obj.history    = zeros([obj.ITERMAX 1]);
            if isfield(newlimit, 'timemax')
                obj.TIMEMAX    = newlimit.timemax;
            end
            obj.WALLMAX    = elapsed.wallMax;
            obj.iteration  = elapsed.iteration;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
            obj.timePercent = 100*obj.time/obj.TIMEMAX;
            obj.updateWallTime();
            obj.dtAverage  = mean(obj.history);
            
            
        end
        
    end%PUBLIC
    
    %===============================================================================================
    methods (Access = private) %                                                P R I V A T E    [M]
        
        
        %_____________________________________________________________________________ appendHistory
        % Appends a new dTime value to the history.
        function appendHistory(obj)
            if length(obj.history) < obj.iteration
                obj.history = [obj.history; zeros(obj.ITERMAX-length(obj.history),1)];
            end
            if obj.iteration > 0
                obj.history(obj.iteration) = 2*obj.dTime;
            end
        end
        
    end%PROTECTED
    
    %===============================================================================================
    methods (Static = true) %                                                     S T A T I C    [M]
        
        %_______________________________________________________________________________ getInstance
    end%STATIC
    
end%CLASS
