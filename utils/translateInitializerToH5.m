function translateInitializerToH5(ini, fname)

h5create(fname,'/placeholder',1,'Datatype','single');


h5writeatt(fname, '/', 'cfl', ini.cfl);
h5writeatt(fname, '/', 'd3h', ini.geometry.d3h);
h5writeatt(fname, '/', 'globalResolution', int32(ini.geometry.globalDomainRez));
h5writeatt(fname, '/', 'iterMax', int32(ini.iterMax));
h5writeatt(fname, '/', 'multifluidDragMethod', int32(ini.multifluidDragMethod));
h5writeatt(fname, '/', 'timeMax', int32(ini.timeMax));
%h5writeatt(fname, '/', );

writeSimpleStruct(fname, '/frameParameters', ini.frameParameters);

writeSimpleStruct(fname, '/fluidDetail1', ini.fluidDetails(1));
B = BCManager();
bs = B.expandBCStruct(ini.bcMode{1});

q = @B.bcModeToNumber;
bc1 = [q(bs{1,1}) q(bs{2,1})  q(bs{1,2}) q(bs{2,2})   q(bs{1,3}) q(bs{2,3})];
h5writeatt(fname, '/fluidDetail1', 'bcmodes', int32(bc1));

if ini.numFluids > 1
    writeSimpleStruct(fname, '/fluidDetail2', ini.fluidDetails(2));
    bs = B.expandBCStruct(ini.bcMode{2});
    bc1 = [q(bs{1,1}) q(bs{2,1})  q(bs{1,2}) q(bs{2,2})   q(bs{1,3}) q(bs{2,3})];
    h5writeatt(fname, '/fluidDetail2', 'bcmodes', int32(bc1));
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