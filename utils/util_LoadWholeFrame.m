function massiveFrame = util_loadWholeFrame(basename, padding, framenum, precise)

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

massiveFrame = frame;
globalRes = frame.parallel.globalDims;

beCarefulWithRZ = 0;
if (globalRes(3) > 1) && (globalRes(2) == 1)
    beCarefulWithRZ = 1;
    % I'm not sure if it's my export or NetCDF that effs this up,
    % but we have to be careful with this case
end

massiveFrame.myOffset = [0 0 0];

massiveFrame.mass = zeros(globalRes, precise);
massiveFrame.momX = zeros(globalRes, precise);
massiveFrame.momY = zeros(globalRes, precise);
massiveFrame.momZ = zeros(globalRes, precise);
massiveFrame.ener = zeros(globalRes, precise);
if numel(frame.magX) > 1 % Having it as zero means we have a scalar if something expects B to be there
    massiveFrame.magX = zeros(globalRes, precise);
    massiveFrame.magY = zeros(globalRes, precise);
    massiveFrame.magZ = zeros(globalRes, precise);
else
    massiveFrame.magX = 0;
    massiveFrame.magY = 0;
    massiveFrame.magZ = 0;
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
    if numel(frame.magX) > 1
        massiveFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks);
        massiveFrame.magY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magY, ranks);
        massiveFrame.magZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magZ, ranks);
    end
    
    if u == numel(ranks); break; end
    u=u+1;
    frame = util_LoadFrameSegment(basename, padding, ranks(u), framenum);
end

end % function

function y = trimHalo(x, nprocs)
  U = 1:size(x,1); V = 1:size(x,2); W = 1:size(x,3);

  if size(nprocs,1) > 1; U = U(4:(end-3)); end
  if size(nprocs,2) > 1; V = V(4:(end-3)); end
  if size(nprocs,3) > 1; W = W(4:(end-3)); end

  y = x(U,V,W);
end


