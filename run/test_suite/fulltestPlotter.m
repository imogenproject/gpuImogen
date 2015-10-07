function fulltestPlotter(FTR)
% > FTR : Full Test Result structure saved by the test suite

plotno = 0;
maxplot = 1;
figure();

doquad = input('Plot figures in 2x2 matrix? ');
if doquad; maxplot = 4; end

% Begin plot output
% Tick-tock our little plot creating state machine along as well

% Plot the advection test results
plotno = prepNextPlot(maxplot, plotno);
plotAdvecOutput(FTR.advection.Xalign_mach0);
title('X advection, n = [1 0 0], stationary bg');

plotno = prepNextPlot(maxplot, plotno);
plotAdvecOutput(FTR.advection.Xalign_mach0p5);
title('X advection, n = [1 0 0], moving bg');

plotno = prepNextPlot(maxplot, plotno);
plotAdvecOutput(FTR.advection.XY);
title('Cross-grid advection, n = [4 5 0]');

% Plot the Einfeldt rarefaction test results
plotno = prepNextPlot(maxplot, plotno);
plotEinfeldt(FTR.einfeldt);
title('Einfeldt test results');

% Plot the Sod tube results
plotno = prepNextPlot(maxplot, plotno);
plotSod(FTR.sod.X)
title('Sod tube convergence results');

% Plot centrifuge test results
plotno = prepNextPlot(maxplot, plotno);
plotCentrifuge(FTR.centrifuge);
title('Centrifuge equilibrium maintainence results');

% Plot Sedov-Taylor metric results 
plotno = prepNextPlot(maxplot, plotno);
plotSedov(FTR.sedov3d);
title('3D Sedov-Taylor density errors');

end

function p = prepNextPlot(maxplot, plotno)
p = plotno + 1;
if p > maxplot; figure(); p = 1; end

if maxplot > 1; subplot(2,2,p); end

hold on;

end

function plotAdvecOutput(q)

plot(-log2(q.relativeH), log2(q.err1),'r-x'); % one norm
plot(-log2(q.relativeH), log2(q.err2),'g-x'); % 2 norm
plot(-log2(q.relativeH), .5*(log2(q.err1(1)) + log2(q.err2(1))) + 2*log2(q.relativeH),'k-'); % reference slope of -2

xlabel('-log_2(h * 32)');
ylabel('log_2(metric norms)');

legend(['1-norm, avg slope ' q.L1_Order], ['2-norm, avg slope ' q.L2_Order], 'Reference 2nd order slope');

end

function plotEinfeldt(q)

plot(log2(q.N), log2(q.L1),'r-x');
plot(log2(q.N), log2(q.L2),'g-x');
plot(log2(q.N), .5*(log2(q.L1(1)) + log2(q.L2(1))) - log2(q.N/q.N(1)),'k-');

xlabel('Log_2(1/h)');
ylabel('log_2(metric norms)');

legend('1-Norm','2-Norm','reference -2 slope');

end

function plotSod(q)

plot(log2(q.res), log2(q.L1),'r-x');
plot(log2(q.res), log2(q.L2),'g-x');
plot(log2(q.res), .5*(log2(q.L1(1))+log2(q.L2(1))) - 1*log2(q.res/q.res(1)),'k-');

xlabel('log_2(1/h)');
ylabel('log_2(metric norms');

legend('1-Norm','2-Norm','slope of -1');

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

%legend(leg, 'Location','EastOutside')

% Plot the t=end metrics to show convergence of final solution
plot(1:N,log2(q.L1(:,end)),'r-x');
plot(1:N,log2(q.L2(:,end)),'b-o');

end

function plotSedov(q)

plot(q.times, q.rhoL1,'-x');
plot(q.times, q.rhoL2,'-x');

xlabel('Simulation time (end = r \rightarrow 0.45)')
ylabel('Density error norm');

end
