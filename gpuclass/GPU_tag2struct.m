function s = GPU_tag2struct(t)
% Accept a GPU int64[] tag and convert it to a human-readable struct.
% cudaCommon.h enumarates what the indices refer to

partdir = '';
switch(double(t(6)) );
	case 1; partdir = 'X';
	case 2; partdir = 'Y';
	case 3; partdir = 'Z';
end

memlay = '';
switch(double(t(9)));
    case 1; memlay = 'XYZ';
    case 2; memlay = 'XZY';
    case 3; memlay = 'YXZ';
    case 4; memlay = 'YZX';
    case 5; memlay = 'ZXY';
    case 6; memlay = 'ZYX';
end

b = (numel(t) - 9)/2;

x = reshape(t(10:end), [2 b]);

s = struct('arrayDimensions', double(t(1:3)'), ...
           'numel', prod(double(t(1:3))), ...
           'haloSize', double(t(5)), ...
           'exteriorHalos', double(t(8)), ...
           'partitionDirection', partdir, ...
           'memoryLayout', memlay, ...
           'numGPUs', double(t(7)), ...
           'slabInfo', double(t(4)), ...
           'pointers', x);

end
