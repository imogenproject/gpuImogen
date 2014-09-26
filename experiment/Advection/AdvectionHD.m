classdef AdvectionHD < handle
% Class annotation template for creating new classes.
%___________________________________________________________________________________________________ 

%===================================================================================================
        properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
        
%===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                           P U B L I C  [P]
    
    rho0;

    times;
    amps;

    A0 = .01;

    end %PUBLIC

%===================================================================================================
    properties (SetAccess = protected, GetAccess = protected) %                P R O T E C T E D [P]
    end %PROTECTED
    
%===================================================================================================
    methods %                                                                     G E T / S E T  [M]
        end%GET/SET
        
%===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]

    function obj = AdvectionHD(run, mass, mom, ener, mag)
        obj.rho0 = mass.array;
    end

    function FrameAnalyzer(obj, run, mass, mom, ener, mag)

    obj.times(end+1) = sum(run.time.history);

    q = fft(mass.array(:,1));
    obj.amps(:,end+1) = q(2:11);

    w0 = 2*pi*sqrt(5/3);

    figure(1);
    hold off;
    plot(obj.times, abs(obj.amps(2,:)/512));
hold on;
plot(obj.times, obj.A0^2*w0*obj.times,'-x')
    pause(.01);

    end

    function finish(obj, run)
save('hdoutput.mat','obj');
    end
        end%PUBLIC
        
%===================================================================================================        
        methods (Access = protected) %                                      P R O T E C T E D    [M]
        end%PROTECTED
                
%===================================================================================================        
        methods (Static = true) %                                                 S T A T I C    [M]
        end%PROTECTED
        
end%CLASS
