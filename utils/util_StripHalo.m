function frameOut = util_StripHalo(frameIn)

frameOut = frameIn;

h = frameIn.parallel.haloAmt;

a = 1; if bitand(frameIn.parallel.halobits, 1); a = 1+h; end
b = size(frameIn.mass, 1); if bitand(frameIn.parallel.halobits, 2); b = b - h; end
ix = a:b;

a = 1; if bitand(frameIn.parallel.halobits, 4); a = 1+h; end
b = size(frameIn.mass, 2); if bitand(frameIn.parallel.halobits, 8); b = b - h; end
iy = a:b;

a = 1; if bitand(frameIn.parallel.halobits, 16); a = 1+h; end
b = size(frameIn.mass, 3); if bitand(frameIn.parallel.halobits, 32); b = b - h; end
iz = a:b;

fields = {'mass','momX','momY','momZ','ener'};
for x = 1:5
    fld = fields{x};
    frameOut.(fld) = frameIn.(fld)(ix,iy,iz);
end

if isfield(frameIn, 'mass2')
    fields = {'mass2','momX2','momY2','momZ2','ener2'};
    for x = 1:5
        fld = fields{x};
        frameOut.(fld) = frameIn.(fld)(ix,iy,iz);
    end
end

if numel(frameIn.magX) > 1
    fields = {'magX','magY','magZ'};
    for x = 1:3
        fld = fields{x};
        frameOut.(fld) = frameIn.(fld)(ix,iy,iz);
    end
end

end
