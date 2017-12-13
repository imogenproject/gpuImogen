function fulltestPlotter(FTR, doquad)
% > FTR : Full Test Result structure saved by the test suite

plotno = 1;
maxplot = 1;

if nargin < 2
    doquad = input('Plot figures in 2x2 matrix? ');
end
if doquad; maxplot = 4; end

% Begin plot output
% Tick-tock our little plot creating state machine along as well

% Plot the advection test results
if isfield(FTR,'advection')
    if isfield(FTR.advection,'Xalign_mach0')
        plotno = prepNextPlot(maxplot, plotno);
        plotAdvecOutput(FTR.advection.Xalign_mach0);
        title('X advection, n = [1 0 0], stationary bg,');
        stylizePlot(gca());
    end
    
    if isfield(FTR.advection, 'Xalign_mach0p5')
        %plotno = prepNextPlot(maxplot, plotno);
        plotAdvecOutput(FTR.advection.Xalign_mach0p5);
        title('X advection, n = [1 0 0], moving bg');
        stylizePlot(gca());
    end
    
    if isfield(FTR.advection, 'XY')
        plotno = prepNextPlot(maxplot, plotno);
        plotAdvecXYOutput(FTR.advection.XY);
        title('Cross-grid advection, n = [4 5 0]');
        stylizePlot(gca());
    end
end

if isfield(FTR,'einfeldt')
    % Plot the Einfeldt rarefaction test results
    plotno = prepNextPlot(maxplot, plotno);
    plotEinfeldt(FTR.einfeldt);
    title('Einfeldt test results');
    stylizePlot(gca());
end

if isfield(FTR,'sod')
    % Plot the Sod tube results
    plotno = prepNextPlot(maxplot, plotno);
    plotSod(FTR.sod.X)
    title('Sod tube convergence results');
    stylizePlot(gca());
end

if isfield(FTR,'noh')
    % Plot the Sod tube results
    plotno = prepNextPlot(maxplot, plotno);
    plotNoh(FTR.noh.X)
    title('Noh tube convergence results');
    stylizePlot(gca());
end

if isfield(FTR,'doubleBlast')
    % Plot the double blast refinement results
    plotno = prepNextPlot(maxplot, plotno);
    plotDoubleBlast(FTR.doubleBlast);
    title('Double blast test convergence results');
    stylizePlot(gca());
end

if isfield(FTR,'centrifuge')
    % Plot centrifuge test results
    plotno = prepNextPlot(maxplot, plotno);
    plotCentrifuge(FTR.centrifuge);
    title('Centrifuge equilibrium maintainence results');
    stylizePlot(gca());
end

if isfield(FTR, 'sedov3d')
    % Plot Sedov-Taylor metric results
    plotno = prepNextPlot(maxplot, plotno);
    plotSedov(FTR.sedov3d);
    title('3D Sedov-Taylor density errors');
    stylizePlot(gca());
end

if isfield(FTR, 'dustybox')
    % Plot dusty box results
    plotno = prepNextPlot(maxplot, plotno);
    plotDustybox(FTR.dustybox);
    title('Accuracy of dustybox solutions');
    stylizePlot(gca());
end

end

function p = prepNextPlot(maxplot, plotno)
p = plotno + 1;
if p > maxplot;
    x = figure();
    x.Position = [x.Position(1:2) 700 400];
    p = 1;
end

if maxplot > 1; subplot(2,2,p); end

hold on;

end

function stylizePlot(A)

A.XLabel.FontSize = 14;
A.YLabel.FontSize = 14;
A.ZLabel.FontSize = 14;

end

function plotAdvecOutput(q)

plot(log2(q.N), log2(q.L1),'r-x'); % one norm
plot(log2(q.N), log2(q.L2),'g-x'); % 2 norm
plot(log2(q.N), .5*(log2(q.L1(1)) + log2(q.L2(1))) - 2*log2(q.N/q.N(1)),'k-'); % reference slope of -2

xlabel('log_2(# pts)');
ylabel('log_2[|\rho - \rho_{exact}|/|\delta \rho_{ini}|]');

legend(['1-norm, avg slope ' num2str(q.L1_Order)], ['2-norm, avg slope ' num2str(q.L2_Order)], 'Reference 2nd order slope');

end

function plotAdvecXYOutput(q)

end

function plotDustybox(q)

shiftA = 0;%floor(log2(abs(q.mid.L2(1)/q.slow.L2(1))));
shiftB = 0;%ceil(log2(abs(q.mid.L2(1)/q.supersonic.L2(1))));

plot(log2(q.slow.N), shiftA + log2(abs(q.slow.L2)),'b-x');
plot(log2(q.mid.N), log2(abs(q.mid.L2)),'g-x');
plot(log2(q.supersonic.N), shiftB + log2(abs(q.supersonic.L2)),'r-x');

z = ones(size(q.mid.N));

plot(log2(q.mid.N), -16.61*z,'k-');
plot(log2(q.mid.N), -29.9*z,'k-.');

xlabel('log_2(# pts)');
ylabel('Velocity error at t=5');

legend('M_{ini} = .01', 'M_{ini} = .25', 'M_{ini} = 2.0', '10^{-5} error','10^{-9} error');

end

function plotDustywave(q)

end

function plotEinfeldt(q)

plot(log2(q.N), log2(q.L1),'r-x');
plot(log2(q.N), log2(q.L2),'g-x');
plot(log2(q.N), .5*(log2(q.L1(1)) + log2(q.L2(1))) - log2(q.N/q.N(1)),'k-');

xlabel('log_2(# pts)');
ylabel('log_2(|\rho - \rho_{exact}|)');

legend('1-Norm','2-Norm','reference -1 slope');

end

function plotSod(q)

plot(log2(q.N), log2(q.L1),'r-x');
plot(log2(q.N), log2(q.L2),'g-x');
plot(log2(q.N), .5*(log2(q.L1(1))+log2(q.L2(1))) - 1*log2(q.N/q.N(1)),'k-');

xlabel('log_2(# pts)');
ylabel('log_2(|\rho - \rho_{exact}|)');

legend('1-Norm','2-Norm','Reference slope of -1');

end

function plotNoh(q)

plot(log2(q.N), log2(q.L1),'r-x');
plot(log2(q.N), log2(q.L2),'g-x');
plot(log2(q.N), .5*(log2(q.L1(1))+log2(q.L2(1))) - 1*log2(q.N/q.N(1)),'k-');

xlabel('log_2(# pts)');
ylabel('log_2(|\rho - \rho_{ref}|)');

legend('1-Norm','2-Norm','Reference slope of -1');

end

function plotDoubleBlast(q)
plot(log2(q.N), log2(q.L1),'r-x');
plot(log2(q.N), log2(q.L2),'g-x');
plot(log2(q.N), 2+.5*(log2(q.L1(1))+log2(q.L2(1))) - 1*log2(q.N/q.N(1)),'k-');
xlabel('log_2(# pts)');
ylabel('|\rho - \rho_{max refinement}|');

legend('1-norm','2-norm','Reference slope of -1');
end

function plotCentrifuge(q)

%plot(q.T', log2(q.L1)','-x');
%plot(q.T', log2(q.L2)','-.o');

grid on;

%xlabel('Dynamic times');
%ylabel('log_2(metric(\delta \rho))');

% Number of sims that were run
N = size(q.T,1);

% Construct the big list of legend entries
leg = cell(2*N,1);
leg{1} = '1-norm, lowest resolution';
leg{N+1} = '2-norm, lowest resolution';

for k = 2:(N-1);
  leg{k} = '.';
  leg{N+k}='.';
end

if N > 1;
  leg{N} = '1-norm, highest resolution';
  leg{2*N}='2-norm, highest resolution';
end

legend(leg, 'Location','EastOutside')

% Plot the t=end metrics to show convergence of final solution
plot(3+(1:N),log2(q.L1(:,end)),'r-x');
plot(3+(1:N),log2(q.L2(:,end)),'b-o');

legend('L_1 norm','L_2 norm');
xlabel('log_2(# pts)');
ylabel('log_2(|\rho - \rho_{ini}|)');

end

function plotSedov(q)

plot(q.times, q.rhoL1,'-x');
plot(q.times, q.rhoL2,'-o');

xlabel('Simulation time (end = r \rightarrow 0.45)')
ylabel('Norm(\rho - \rho_{exact})');

end

