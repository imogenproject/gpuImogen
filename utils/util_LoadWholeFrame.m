function bigFrame = util_loadWholeFrame(basename, padding, framenum, precise)

if (nargin < 4) || (precise == 1); precise = 'double'; else; precise = 'single'; end

if isa(basename,'double')
    strnames={'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
    try
        basename = strnames{basename};
    catch MERR
        basename
        error('Numeric basename passed was not an integer [1, 7].');
    end
end

frame = util_LoadFrameSegment(basename, padding, 0, framenum); % We need one for reference

bigFrame = frame;
globalRes = frame.parallel.globalDims;

beCarefulWithRZ = 0;
if (globalRes(3) > 1) && (globalRes(2) == 1)
    beCarefulWithRZ = 1;
    % I'm not sure if it's my export or NetCDF that effs this up,
    % but we have to be careful with this case
end

bigFrame.myOffset = [0 0 0];

bigFrame.mass = zeros(globalRes, precise);
bigFrame.momX = zeros(globalRes, precise);
bigFrame.momY = zeros(globalRes, precise);
bigFrame.momZ = zeros(globalRes, precise);
bigFrame.ener = zeros(globalRes, precise);
if numel(frame.magX) > 1 % Having it as zero means we have a scalar if something expects B to be there
    bigFrame.magX = zeros(globalRes, precise);
    bigFrame.magY = zeros(globalRes, precise);
    bigFrame.magZ = zeros(globalRes, precise);
else
    bigFrame.magX = 0;
    bigFrame.magY = 0;
    bigFrame.magZ = 0;
end

ranks = frame.parallel.geometry;
fieldset = {'mass','momX','momY','momZ','ener'};
bset     = {'magX','magY','magZ'};

u = 1;

while u <= numel(ranks)
    fs = size(frame.mass);
    if beCarefulWithRZ;
        for N = 1:5; frame.(fieldset{N}) = reshape(frame.(fieldset{N}),[fs(1) 1 fs(2)]); end
        if numel(frame.magX) > 1
            for N = 1:3; frame.(bset{N}) = reshape(frame.(bset{N}),[fs(1) 1 fs(2)]); end
        end
    end
    
    fs = size(frame.mass); if numel(fs) == 2; fs(3) = 1;  end
    rs = size(ranks); if numel(rs) == 2; rs(3) = 1; end
    frmsize = fs - 2*frame.haloAmt*(rs > 1);
    if numel(frmsize) == 2; frmsize(3) = 1; end

    frmset = {frame.parallel.myOffset(1)+(1:frmsize(1)), ...
              frame.parallel.myOffset(2)+(1:frmsize(2)), ...
              frame.parallel.myOffset(3)+(1:frmsize(3))};

    bigFrame.mass(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.mass, ranks, frame.haloAmt);
    bigFrame.momX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momX, ranks, frame.haloAmt);
    bigFrame.momY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momY, ranks, frame.haloAmt);
    bigFrame.momZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momZ, ranks, frame.haloAmt);
    bigFrame.ener(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.ener, ranks, frame.haloAmt);
    if numel(frame.magX) > 1
        bigFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks, frame.haloAmt);
        bigFrame.magY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magY, ranks, frame.haloAmt);
        bigFrame.magZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magZ, ranks, frame.haloAmt);
    end
    
    if u == numel(ranks); break; end
    u=u+1;
    frame = util_LoadFrameSegment(basename, padding, ranks(u), framenum);
end

end % function

function y = trimHalo(x, nprocs, haloamt)
  U = 1:size(x,1); V = 1:size(x,2); W = 1:size(x,3);

  if size(nprocs,1) > 1; U = U((haloamt+1):(end-haloamt)); end
  if size(nprocs,2) > 1; V = V((haloamt+1):(end-haloamt)); end
  if size(nprocs,3) > 1; W = W((haloamt+1):(end-haloamt)); end

  y = x(U,V,W);
end


