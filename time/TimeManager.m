classdef TimeManager < handle
% The manager class responsible for handling time related variables and operations. This is a 
% singleton class to be accessed using the getInstance() method and not instantiated directly.

%===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
%===================================================================================================
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
        updateMode;  % Frequency of updates (PER_ITERATION or PER_STEP)              int
        timePercent; % Percent complete based on simulation time.                    double
        iterPercent; % Percent complete based on iterations of maximum.              double
        wallPercent; % Percent complete based on wall time.                          double
        running;     % Specifies if the simulation should continue running.          logical
        dtAverage;   % The accumulated mean timestep                                 double

        firstWallclockValue;
    end%PUBLIC
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = private) %                          P R I V A T E [P]
        parent;      % Parent manager                                                ImogenManager
    end %PRIVATE

%===================================================================================================
    methods %                                                                      G E T / S E T  [M]
        
        function result = get.running(obj)
            result = (obj.iteration <= obj.ITERMAX) && (obj.time < obj.TIMEMAX) ...
                      && (obj.wallTime < obj.WALLMAX);
        end
        
    end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    
	function recordWallclock(obj)
            obj.firstWallclockValue = clock;
        end

%___________________________________________________________________________________________________ update
% Calculate the correct timestep for the upcoming flux iteration (both the forward and backward 
% steps) of the main loop according to the Courant-Freidrichs-Levy condition for a magnetic fluid.
%
%>< run             Imogen run manager                                         ImogenManager
%>< mass            mass density array                                         FluidArray
%>< mom             momentum density array                                     FluidArray(3)
%>< ener            energy density array                                       FluidArray        
%>< mag             magnetic field array                                       MagnetArray(3)
        function update(obj, mass, mom, ener, mag, fluxDirection)
        
            %--- Initialization ---%
            
            % Skips calculating the timestep if this is the second portion of the step and the 
            % timestep update mode is one per iteration and not every step.
%            if (fluxDirection > 1 && obj.updateMode ~= ENUM.TIMEUPDATE_PER_STEP)
%                return; 
%            end

            cmax        = 1e-2; % Min threshold to prevent excessively large timesteps.
            gridIndex   = 0;    % Defaults max velocity direction to invalid error value.

            %--- Find max velocity ---%
            %           Find the maximum fluid velocity in the grid and its vector direction.
            if obj.parent.pureHydro == 1
                soundSpeed = cudaSoundspeed(mass, ener, mom(1), mom(2), mom(3), obj.parent.GAMMA);
            else
                soundSpeed = cudaSoundspeed(mass, ener, mom(1), mom(2), mom(3), ...
                             mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, obj.parent.GAMMA);
            end

            [cmax gridIndex] = directionalMaxFinder(mass, soundSpeed, mom(1), mom(2), mom(3));
            GPU_free(soundSpeed);
            % Communicate the maximum cfl-determining limit to our comrades in arms
            % FIXME: Make this use mpi_reduce (and implement an interface to it in /mpi)
            % WARNING: Well it seems that nowhere is idx actually /used/, so I'm dropping
            % WARNING: it until a use pops up.
%            cmaxB       = mpi_max(cmax);
%if mpi_amirank0(); disp(cmaxB); end
            cmax       = mpi_allgather(cmax);
%if mpi_amirank0(); disp(cmax'); end
            gridIndex  = mpi_allgather(gridIndex);
            [cmax idx] = max(cmax);
            gridIndex  = gridIndex(idx);

            %--- Check for errors ---%
            % If the gridIndex isn't valid then a CFL error is the cause, which means that the 
            % propagation over the last step was too large and has corrupted the run.
            if ~isreal(cmax)
%(gridIndex < 1)
                fireCFLError(run, cmax); 
                gridIndex = 1; 
            end
 
            %--- Calculate new timestep ---%
            %           Using Courant-Freidrichs-Levy (CFL) condition determine safe step size
            %           accounting for maximum simulation time.
            obj.dTime = obj.CFL*obj.parent.MINDGRID(gridIndex)/cmax;
            newTime   = obj.time + 2*obj.dTime; % Each individual fwd or bkwd sweep is a full step in time
            if (newTime > obj.TIMEMAX)
                obj.dTime = .5*(obj.TIMEMAX - obj.time);
            end
            obj.timePercent = 100*newTime/obj.TIMEMAX;
        end

%___________________________________________________________________________________________________ updateUI
        function updateUI(obj)
            save = obj.parent.save;
            
            %--- Clock first loops ---%
            %           Clocks the first loop of execution and uses that to determine an estimated
            %           time to complete that is displayed in the UI and log files.
            if obj.iteration < 5
                switch obj.iteration 
                    
                    case 1;        %Activate clock timer for the first loop        
                        tic;
                        
                    case 4;        %Stop clock timer and use the elapsed time to predict total run time
                        tPerStep = toc/3;
                        save.logPrint('\tFirst three timesteps averaged %0.4g secs ea.\n', tPerStep);

                        if (obj.iterPercent > obj.timePercent)
                            secsRemaining = tPerStep*(obj.ITERMAX-3);
                        else
                            secsRemaining = tPerStep*ceil(obj.TIMEMAX/obj.time);
                        end

                        finTime     = now + secsRemaining/86400;
                        expComplete = datestr( finTime , 'HH:MM:SS PM');

                        if ( floor(finTime) - floor(now) >= 1.0 )
                            expComplete = [expComplete ' on ' datestr( finTime, 'mmm-dd')];
                        end
                        save.logPrint('\tEst. time of completion: %s\n', expComplete);

                        dDays       = floor(secsRemaining/86400); rem = secsRemaining - 86400*dDays;
                        dHours      = floor( rem / 3600 );        rem = rem - 3600*dHours;
                        dMinutes    = floor( rem / 60 );          rem = rem - 60*dMinutes;
                        dSecs       = ceil(rem);
                        save.logPrint('\tEst. wallclock compute time: [%g days | %g hr | %g mins | %g sec ]\n', ...
                                      dDays, dHours, dMinutes, dSecs);
                end
            end

            %--- Update UI for critical loops ---%
            %           Displays information to the UI for critical points during the run.
            if (save.updateUI)
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
                curTime = strcat(datestr(cTime , 'HH:MM:SS PM'),' on', datestr(cTime, ' mm-dd-yy'));
                save.logPrint('[[ %0.3g%% | %s |  %s ]]\n', compPer, infoStr, curTime);
            end

            obj.parent.abortCheck();
            
        end

%___________________________________________________________________________________________________ updateWallTime
        function updateWallTime(obj)
            obj.wallTime    = etime(clock(), obj.startTime)/3600;
            obj.wallPercent = 100*obj.wallTime/obj.WALLMAX;
        end
        
%___________________________________________________________________________________________________ step
% Increments the iteration variable by one for the next loop.
        function step(obj)
            obj.iteration   = obj.iteration + 1;
            obj.updateUI();
            obj.time = obj.time + 2*obj.dTime;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
            obj.appendHistory();

        end
                
%___________________________________________________________________________________________________ toStruct
% Converts the TimeManager object to a structure for saving and non-class use.
% # result        The structure resulting from conversion of the TimeManager object.                        Struct
        function result = toStruct(obj)
            result.time       = obj.time;
            result.history    = obj.history(1:(obj.iteration+1));
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
            obj.history    = elapsed.history;
            obj.ITERMAX    = newlimit.itermax;
            obj.TIMEMAX    = newlimit.timemax;
            obj.WALLMAX    = elapsed.wallMax;
            obj.iteration  = elapsed.iteration;
            obj.iterPercent = 100*obj.iteration/obj.ITERMAX;
            obj.timePercent = 100*obj.time/obj.TIMEMAX;
            obj.updateWallTime();
            obj.dtAverage  = mean(obj.history);
        end 

    end%PUBLIC 
        
%===================================================================================================
        methods (Access = private) %                                            P R I V A T E    [M]
                
%___________________________________________________________________________________________________ TimeManager
% Creates a new TimeManager instance.
        function obj = TimeManager() 
            obj.startTime   = clock;
            obj.time        = 0;
            obj.dTime       = 0;
            obj.iteration   = 0;
            obj.ITERMAX     = 1000;
            obj.TIMEMAX     = 5000;
            obj.WALLMAX     = 1e5;
            obj.history     = zeros(obj.ITERMAX,1);
            obj.updateMode  = ENUM.TIMEUPDATE_PER_ITERATION;
            obj.iterPercent = 0;
            obj.timePercent = 0;
            obj.wallPercent = 0;
            obj.dtAverage   = 0;
        end
                
%___________________________________________________________________________________________________ appendHistory
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

%===================================================================================================
        methods (Static = true) %                                                 S T A T I C    [M]
                
%___________________________________________________________________________________________________ getInstance
% Accesses the singleton instance of the TimeManager class, or creates one if none have
% been initialized yet.
                function singleObj = getInstance()
                        persistent instance;
                        if isempty(instance) || ~isvalid(instance) 
                                instance = TimeManager();
                        end
                        singleObj = instance;
                end
          
        end%STATIC
        
end%CLASS
