function translateInitializerToH5(ini, outfile, savefilePrefix, saveByTime)

h5create(outfile,'/placeholder',1,'Datatype','single');

h5writeatt(outfile, '/', 'savefilePrefix', savefilePrefix);

h5writeatt(outfile, '/', 'cfl', ini.cfl);
h5writeatt(outfile, '/', 'd3h', ini.geometry.d3h);
h5writeatt(outfile, '/', 'globalResolution', int32(ini.geometry.globalDomainRez));
h5writeatt(outfile, '/', 'iterMax', int32(ini.iterMax));
h5writeatt(outfile, '/', 'multifluidDragMethod', int32(ini.multifluidDragMethod));
h5writeatt(outfile, '/', 'timeMax', ini.timeMax);
%h5writeatt(fname, '/', );

s = struct('slice', ini.slice, 'percent', ini.ppSave, 'bytime', int32(saveByTime ~= 0));
writeSimpleStruct(outfile, '/save', s);

writeSimpleStruct(outfile, '/frameParameters', ini.frameParameters);

writeSimpleStruct(outfile, '/fluidDetail1', ini.fluidDetails(1));
B = BCManager();
bs = B.expandBCStruct(ini.bcMode{1});

q = @B.bcModeToNumber;
bc1 = [q(bs{1,1}) q(bs{2,1})  q(bs{1,2}) q(bs{2,2})   q(bs{1,3}) q(bs{2,3})];
h5writeatt(outfile, '/fluidDetail1', 'bcmodes', int32(bc1));

if ini.numFluids > 1
    writeSimpleStruct(outfile, '/fluidDetail2', ini.fluidDetails(2));
    bs = B.expandBCStruct(ini.bcMode{2});
    bc1 = [q(bs{1,1}) q(bs{2,1})  q(bs{1,2}) q(bs{2,2})   q(bs{1,3}) q(bs{2,3})];
    h5writeatt(outfile, '/fluidDetail2', 'bcmodes', int32(bc1));
end


end

function writeSimpleStruct(fname, sname, s)
% create a dataset to be our placeholder
h5create(fname, sname, 1, 'DataType', 'single');

% writes all in native type
q = fieldnames(s);
for x = 1:numel(q)
    h5writeatt(fname, ['/' sname], [q{x}], s.(q{x}));
end

end