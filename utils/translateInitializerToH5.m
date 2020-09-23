function translateInitializerToH5(ini, outfile, savefilePrefix, saveByTime)

h5create(outfile,'/placeholder',1,'Datatype','single');

h5writeatt(outfile, '/', 'savefilePrefix', savefilePrefix);

h5writeatt(outfile, '/', 'cfl', ini.cfl);
h5writeatt(outfile, '/', 'd3h', ini.geometry.d3h);
h5writeatt(outfile, '/', 'globalResolution', int32(ini.geometry.globalDomainRez));
h5writeatt(outfile, '/', 'iterMax', int32(ini.iterMax));
h5writeatt(outfile, '/', 'multifluidDragMethod', int32(ini.multifluidDragMethod));
h5writeatt(outfile, '/', 'timeMax', ini.timeMax);
h5writeatt(outfile, '/', 'iniFrame', int32(0));

% Geometry mode and frame rotation support
cb = ini.geometry.circularBCs;
cmodes = cb(1) + 2*cb(2) + 4*cb(3);
geom = struct('geomType', int32(ini.geometry.pGeometryType), 'innerRadius', ...
 ini.geometry.pInnerRadius, 'x0', ini.geometry.affine, 'd3h', ini.geometry.d3h, 'frameCenter', ...
 ini.frameParameters.rotateCenter, 'frameOmega', ini.frameParameters.omega, 'circularity', int32(cmodes));
writeSimpleStruct(outfile, '/geometry', geom);

s = struct('slice', ini.slice, 'percent', [ini.ppSave.dim1 ini.ppSave.dim2 ini.ppSave.dim3], ...
 'bytime', int32(saveByTime ~= 0));
writeSimpleStruct(outfile, '/save', s);

writeSimpleStruct(outfile, '/frameParameters', ini.frameParameters);

% Multifluid drag support is embedded in the gas fluid model
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

% Gravity support:
% /gravConstant is written by experiment/Initializer.m:314 because it needs data we don't have here

% Radiation support:
s = struct('theta', ini.radiation.exponent, 'beta', ini.radiation.setStrength, 'minTemp', ini.Tcutoff);
if strcmp(ini.radiation.strengthMethod, 'preset') == 0
    disp('WARNING: Radiation strength method is not "preset" and this is the only method supported by imogenCore.\n');
end

writeSimpleStruct(outfile, '/radiation', s);

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
