classdef RHD_Analyzer < SimulationAnalyzer

    properties(SetAccess = public, GetAccess = public)
        lastLinearFrame;
    end

    properties(SetAccess = public, GetAccess = public)
        nModes; % number of transverse modes in [Y Z] direction
        nFrames;
        is2d;   % True if run was of z extent 1, otherwise false.

        linearFrames;

        frontX; % Traces shock front X position at all times

        rmsdrho, rmsdP; % Traces rms values of transverse fluctuations of density and pressure
        rhobar, vxbar, vybar, Pbar; % Traces ky=0 parts of state (radial component)
    end

    properties(SetAccess = protected, GetAccess = protected)
    end

methods % SET

end

methods (Access = public)

    function obj = RHD_Analyzer(basename, framerange,  verbosity)
        if nargin < 3;
            disp('Did not get enough arguments to run automatically (to do this RHD_Analyzer has to be created with the form RHD_Analyzer(''basename'', [frame:range], 1=verbose/0=silent); e.g. rad = RHD_Analyzer(''2D_XY'', [0:25:100], 1);). Please input them below:');
            basename = input('Base name of savefiles to analyze. Typically either ''2D_XY'' or ''3D_XYZ'':', 's');
            framerange = input('First set of saveframe #s to analyze; Empty is OK: ');
            verbosity = input('1 to have a * printed per frame analyzed, 0 to run more quietly:');
        end

        obj = obj@SimulationAnalyzer(basename, framerange, verbosity);
        obj.setAnalyzerFunction(@obj.FrameAnalyzer);

        obj.addFields({'frontX','rmsdrho','rmsdP','rhobar','vxbar','vybar','Pbar'});

        if numel(framerange) > 1; obj.updateAnalysis(); end
    end

    function help(obj)

disp('There should be help here... I''ll write some once the code is not evolving daily.');
    end
    
    function FrameAnalyzer(obj, frame, time_index)
        % This uses a linear extrapolation to track the shock front's position
        % We define that position as being when density is exactly halfway between analytic equilibrium pre & post values
        % It can remain meaningful into the nonlinear regime as long as the shock's position is still functional in Y and Z.
        dims = size(frame.mass);
        % This is completely stupid since we should just save such information,
        % But we need to determine the shock strength etc here to know what the midpoint mass density ought to be.
        cs = sqrt(frame.gamma*(frame.gamma-1)*(frame.ener(1) - .5*(frame.momX(1).^2+frame.momY(1).^2)./frame.mass(1))/frame.mass(1));
        ms = frame.momX(1)./(cs*frame.mass(1));
        ang = atan(frame.momY(1)/frame.momX(1));
        jump = HDJumpSolver(ms, ang, frame.gamma);

        obj.frontX(time_index,:,:) = reshape(trackFrontHydro(frame.mass, obj.frameX', mean(jump.rho)), [1 size(frame.mass,2) size(frame.mass,3)]);

        N = size(frame.mass,2)*size(frame.mass,3);
        P = (frame.gamma-1)*(frame.ener - .5*(frame.momX.^2+frame.momY.^2)./frame.mass);
        
        obj.rhobar(time_index,:) = mean(frame.mass, 2);
        obj.vxbar(time_index,:)  = mean(frame.momX./frame.mass, 2);
        obj.vybar(time_index,:)  = mean(frame.momY./frame.mass, 2);
        obj.Pbar(time_index,:)   = mean(P, 2);

        for zeta = 1:size(frame.mass,1); % Calculates perturbation by subtracting x-axis quantities as "equilibrium"
            obj.rmsdrho(time_index, zeta) = sum((frame.mass(zeta,:)-obj.rhobar(time_index,zeta)).^2);
            obj.rmsdP  (time_index, zeta) = sum((P(zeta,:)-obj.Pbar(time_index,zeta)).^2);
        end

        obj.rmsdrho(time_index,:) = sqrt(obj.rmsdrho(time_index,:));
        obj.rmsdP(time_index,:)   = sqrt(obj.rmsdP(time_index,:));
    end

    function tau = CI_timefact(obj, ms, dsingular)
        tau = sqrt(5/3)*ms / dsingular;
        return;
    end
    
    function generatePlots(obj)
        t0 = input('Input preferred time normalization factor: ');
        tau = obj.frameT / t0;

        xfrontmean = mean(squeeze(obj.frontX),2);
        
        dfluc = xfrontmean - mean(xfrontmean);
        % Track the locations of zeros
        zeta = dfluc .* circshift(dfluc,1); zeta = zeta(2:(end-1));
        zeta = tau(find(zeta < 0));
        predictW = pi/mean(diff(zeta));
       
        optfunc = @(q, x) q(1)+q(2)*exp(x*q(3)).*cos(x*q(4)+q(5));
        cfit = lsqcurvefit(optfunc, [mean(xfrontmean) 4 .1 predictW 0], tau, xfrontmean);

        % Plot the mean front position over time; This should allow us to track radial oscillations:
        figure();
        plot(tau, xfrontmean-mean(xfrontmean),'r-','linewidth',2);
        hold on;
        plot(tau, optfunc(cfit, tau)-cfit(1),'g*');
        obj.labelplot('Normalized time elapsed', 'Shock front position fluctuation', ...
            sprintf('Fluctuations in average shock front position over time.\nMean position=%f',mean(xfrontmean)));
        grid
        
        annotation(gcf(),'textbox',[.2 .2 .2 .1], 'string',sprintf('Curve fit: %4.3f + %4.3f exp(%4.3f\\tau) cos(%4.3f\\tau+%4.3f)', cfit(1), cfit(2), cfit(3), cfit(4), cfit(5)));

        figure();
        imagesc(obj.frameX, tau, obj.rhobar);
        obj.labelplot('X position','Normalized time','Density waterfall plot');

        figure();
        imagesc(obj.frameX, tau, obj.vybar);
        obj.labelplot('X position','Normalized time','Y velocity waterfall plot');

        figure();
        imagesc(obj.frameX, tau, sqrt(obj.Pbar./obj.rhobar))
        obj.labelplot('X position','Normalized time','sqrt(gamma)*C_s waterfall plot');
    end

end % Public methods

methods % SET

end

methods (Access = protected)

    function labelplot(obj, xlab, ylab, topline)
        xlabel(xlab, 'fontsize',16);
        ylabel(ylab, 'fontsize',16);
        title(topline, 'fontsize', 18);
    end

end % protected methods;

end % class    
