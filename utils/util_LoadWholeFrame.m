function massiveFrame = frame_loadWholeFrame(basename, padding, framenum, verbose)

if nargin < 4; verbose = 0; end

f0 = util_LoadFrameSegment(basename, padding, 0, framenum); % We need one for reference

massiveFrame = f0;
globalRes = f0.parallel.globalDims;

massiveFrame.myOffset = [0 0 0];

massiveFrame.mass = zeros(globalRes);
massiveFrame.momX = zeros(globalRes);
massiveFrame.momY = zeros(globalRes);
massiveFrame.momZ = zeros(globalRes);
massiveFrame.ener = zeros(globalRes);
if numel(f0.magX) > 0
    massiveFrame.magX = zeros(globalRes);
    massiveFrame.magY = zeros(globalRes);
    massiveFrame.magZ = zeros(globalRes);
end

ranks = f0.parallel.geometry;
fieldset = {'mass','momX','momY','momZ','ener'};
bset     = {'magX','magY','magZ'};

  for u = 1:numel(ranks)

    frame = util_LoadFrameSegment(basename, padding, ranks(u), framenum);
    fs = size(frame.mass); if numel(fs) == 2; fs(3) = 1;  end
    rs = size(ranks); if numel(rs) == 2; rs(3) = 1; end
    frmsize = fs - 6*(rs > 1);
    if numel(frmsize) == 2; frmsize(3) = 1; end

    frmset = {frame.parallel.myOffset(1)+(1:frmsize(1)), ...
              frame.parallel.myOffset(2)+(1:frmsize(2)), ...
              frame.parallel.myOffset(3)+(1:frmsize(3))};

    massiveFrame.mass(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.mass, ranks);
    massiveFrame.momX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momX, ranks);
    massiveFrame.momY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momY, ranks);
    massiveFrame.momZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momZ, ranks);
    massiveFrame.ener(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.ener, ranks);
    if numel(f0.magX) > 0
        massiveFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks);
        massiveFrame.magY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magY, ranks);
        massiveFrame.magZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magZ, ranks);
    end
  end

end

function y = trimHalo(x, nprocs)
  y=x;
  if size(nprocs,1) > 1; y = x(4:(end-3),:,:); end
  if size(nprocs,2) > 1; y = y(:,4:(end-3),:); end
  if size(nprocs,3) > 1; y = y(:,:,4:(end-3)); end
end


