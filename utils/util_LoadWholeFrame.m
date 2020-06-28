function bigFrame = util_LoadWholeFrame(prefix, basename, framenum, precise)
% bigFrame = util_LoadWholeFrame(basename, framenum, precise)
% > basename : One of the strings '1D_X', '1D_Y', '1D_Z', '2D_XY', '2D_XZ',
% '2D_XZ', or '3D_XYZ', or an integer from 1 to 7 referring to them in that
% order
    if (nargin < 4) | (precise == 1); precise = 'double'; else; precise = 'single'; end

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

    frame = util_LoadFrameSegment(prefix, basename, 0, framenum); % We need one for reference

    bigFrame = frame;
    globalRes = frame.parallel.globalDims;

    beCarefulWithRZ = 0;
    if (globalRes(3) > 1) && (globalRes(2) == 1)
        beCarefulWithRZ = 1;
        % I'm not sure if it's my export or NetCDF that effs this up,
        % but we have to be careful with this case
    end

    bigFrame.myOffset = [0 0 0];
    
    % Queue off fields in 'frame' to pick names of vars in bigframe
    if isfield(frame, 'momX') % conservative vars
            if isfield(frame, 'mass2')
                dovars= {'mass', 'momX', 'momY', 'momZ', 'ener', 'mass2', 'momX2', 'momY2', 'momZ2', 'ener2'};
            else
                dovars = {'mass', 'momX', 'momY', 'momZ', 'ener'};
            end
        else
            if isfield(frame, 'mass2')
                dovars = {'mass', 'velX', 'velY', 'velZ', 'eint', 'mass2', 'velX2', 'velY2', 'velZ2', 'eint2'};
            else
                dovars = {'mass', 'velX', 'velY', 'velZ', 'eint'};
            end
    end
    
    % Allocate storage
    for q = 1:numel(dovars)
        bigFrame.(dovars{q}) = zeros(globalRes, precise);
    end

    % Waste time on this for some reason
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
    %ranks = [0 2;1 3];
    bset     = {'magX','magY','magZ'};

    u = 1;

    while u <= numel(ranks)
        fs = size(frame.mass);
        if beCarefulWithRZ
            for N = 1:numel(dovars)
                frame.(dovars{N}) = reshape(frame.(dovars{N}),[fs(1) 1 fs(2)]);
            end
            if numel(frame.magX) > 1
                for N = 1:3; frame.(bset{N}) = reshape(frame.(bset{N}),[fs(1) 1 fs(2)]); end
            end
        end

        fs = size(frame.mass); if numel(fs) == 2; fs(3) = 1;  end
        frmsize = fs - frame.parallel.haloAmt*double((bitand(frame.parallel.haloBits, int64([1 4 16])) ~= 0) + (bitand(frame.parallel.haloBits, int64([2 8 32])) ~= 0));
        if numel(frmsize) == 2; frmsize(3) = 1; end

        frmset = {frame.parallel.myOffset(1)+(1:frmsize(1)), ...
                  frame.parallel.myOffset(2)+(1:frmsize(2)), ...
                  frame.parallel.myOffset(3)+(1:frmsize(3))};

        for q = 1:numel(dovars)
            bigFrame.(dovars{q})(frmset{1}, frmset{2}, frmset{3}) = trimHalo(frame.(dovars{q}), frame);
        end

        if numel(frame.magX) > 1
            bigFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, frame);
            bigFrame.magY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magY, frame);
            bigFrame.magZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magZ, frame);
        end
        
        if u == numel(ranks); break; end
        u=u+1;
        frame = util_LoadFrameSegment(prefix, basename, ranks(u), framenum);
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
