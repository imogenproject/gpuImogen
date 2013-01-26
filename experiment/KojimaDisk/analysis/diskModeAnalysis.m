function modes = diskModeAnalysis(frameset, padlen, imgprefix, innercut, noffset)
% Unwraps the disk and projects the azimuthal component into fourier space.
% Return is [nr X 24 modes X nz X ntime]

if nargin < 5; noffset = 0; end
if nargin < 4; innercut = 0; end

frameset = sort(frameset); % Because we all do something stupid now and then

modes = [];

for N = 1:numel(frameset)
  fprintf('Loading frame %i... ',frameset(N));
  cframe = util_LoadWholeFrame('3D_XYZ', padlen, frameset(N));

  fprintf('Azimuthal FFT... ');
  mft = fft(diskUnwrap(cframe.mass),[],2); % Take the azimuthal FFT

  modes(:,:,:,N) = squeeze(mft(:,1:24,:));

save('modeAnalysis.mat','modes');

  fprintf('Printing picture %i...\n', N);

  fig = diskModePlot(cframe, mft, innercut);

  text(.5,.5,sprintf('Time = %.3f mirps',sum(cframe.time.history)/(2*pi)),'HorizontalAlignment','center','fontsize',16);

  set(fig, 'PaperPositionMode', 'auto')
  print(fig,sprintf('%s-%.3i.png',imgprefix,N+noffset),'-dpng', '-r0');

  close(fig);
end

end


