function fig = diskModePlot(cframe, mft, minrad)

resx = 1920;

xd = cframe.parallel.globalDims(1)/2 - minrad;
yd = cframe.parallel.globalDims(3);

aspRatio = yd / xd;

f0 = .47;

resy = aspRatio*2*f0*resx - 60*aspRatio; % For images + labels (~25px each)
if (.5-f0)*resy < 120; resy = resy + 120 -  (.5-f0)*resy;

yfrac = aspRatio*f0*resx / (resy+40);

fig = figure('position',[0 0 resx resy+40]);
dy = 20/(20+resy);

xvals = (1+minrad):cframe.parallel.globalDims(1)/2;

  subplot(2,2,1,'position',[(.25-.5*f0) (.5+dy) f0 yfrac]);
    imagesc(squish(abs(mft(xvals,1,:)))' ); title('Mode 0 amplitude.','fontsize',16);
    axis off; colorbar('EastOutside');
  subplot(2,2,2,'position',[(.75-.5*f0) (.5+dy) f0 yfrac]);
    imagesc(squish(abs(mft(xvals,2,:)))' ); title('Mode 1 amplitude.','fontsize',16);
    axis off; colorbar('EastOutside')
  subplot(2,2,3,'position',[(.25-.5*f0) (dy) f0 yfrac]);
    imagesc(squish(abs(mft(xvals,3,:)))' ); title('Mode 2 amplitude.','fontsize',16);
    axis off; colorbar('EastOutside');
  subplot(2,2,4,'position',[(.75-.5*f0) (dy) f0 yfrac]);
    imagesc(squish(abs(mft(xvals,4,:)))' ); title('Mode 3 amplitude.','fontsize',16);
    axis off; colorbar('EastOutside');

end
