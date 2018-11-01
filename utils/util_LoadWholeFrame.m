function bigFrame = util_LoadWholeFrame(basename, framenum, precise)
% bigFrame = util_LoadWholeFrame(basename, framenum, precise)
% > basename : One of the strings '1D_X', '1D_Y', '1D_Z', '2D_XY', '2D_XZ',
% '2D_XZ', or '3D_XYZ', or an integer from 1 to 7 referring to them in that
% order
    if (nargin < 3) || (precise == 1); precise = 'double'; else; precise = 'single'; end

    if isa(basename,'double')
        strnames={'1D_X','1D_Y','1D_Z','2D_XY','2D_XZ','2D_YZ','3D_XYZ'};
        try
            basename = strnames{basename};
        catch MERR
            disp('Basename is: ');
            disp(basename);
            error('Numeric basename passed was not an integer [1, 7].');
        end
    end

    frame = util_LoadFrameSegment(basename, 0, framenum); % We need one for reference

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
    if isfield(frame,'mass2')
        bigFrame.mass2 = zeros(globalRes, precise);
        bigFrame.momX2 = zeros(globalRes, precise);
        bigFrame.momY2 = zeros(globalRes, precise);
        bigFrame.momZ2 = zeros(globalRes, precise);
        bigFrame.ener2 = zeros(globalRes, precise);
    end

    ranks = frame.parallel.geometry;
    fieldset = {'mass','momX','momY','momZ','ener'};
    bset     = {'magX','magY','magZ'};

    u = 1;

    while u <= numel(ranks)
        fs = size(frame.mass);
        if beCarefulWithRZ
            for N = 1:5; frame.(fieldset{N}) = reshape(frame.(fieldset{N}),[fs(1) 1 fs(2)]); end
            if numel(frame.magX) > 1
                for N = 1:3; frame.(bset{N}) = reshape(frame.(bset{N}),[fs(1) 1 fs(2)]); end
            end
        end

        fs = size(frame.mass); if numel(fs) == 2; fs(3) = 1;  end
        rs = size(ranks); if numel(rs) == 2; rs(3) = 1; end
        frmsize = fs - frame.parallel.haloAmt*double((bitand(frame.parallel.haloBits, int64([1 4 16])) ~= 0) + (bitand(frame.parallel.haloBits, int64([2 8 32])) ~= 0));
        if numel(frmsize) == 2; frmsize(3) = 1; end

        frmset = {frame.parallel.myOffset(1)+(1:frmsize(1)), ...
                  frame.parallel.myOffset(2)+(1:frmsize(2)), ...
                  frame.parallel.myOffset(3)+(1:frmsize(3))};

        bigFrame.mass(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.mass, frame);
        bigFrame.momX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momX, frame);
        bigFrame.momY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momY, frame);
        bigFrame.momZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momZ, frame);
        bigFrame.ener(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.ener, frame);
        if numel(frame.magX) > 1
            bigFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, frame);
            bigFrame.magY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magY, frame);
            bigFrame.magZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magZ, frame);
        end
        
        if isfield(frame,'mass2') % also load secondary plane
            bigFrame.mass2(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.mass2, frame);
            bigFrame.momX2(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momX2, frame);
            bigFrame.momY2(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momY2, frame);
            bigFrame.momZ2(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momZ2, frame);
            bigFrame.ener2(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.ener2, frame);
        end

        if u == numel(ranks); break; end
        u=u+1;
        frame = util_LoadFrameSegment(basename, ranks(u), framenum);
    end

end % function

function y = trimHalo(x, subframe)
    b = subframe.parallel.haloBits;
    h = subframe.parallel.haloAmt;
    ba = (bitand(b, int64([1 2 4 8 16 32])) ~= 0) * 1;

    U = (1+h*ba(1)):(size(x,1)-h*ba(2));
    V = (1+h*ba(3)):(size(x,2)-h*ba(4));
    W = (1+h*ba(5)):(size(x,3)-h*ba(6));
    
    y = x(U,V,W);
end
