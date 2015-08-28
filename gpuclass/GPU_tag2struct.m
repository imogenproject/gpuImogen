function s = GPU_tag2struct(t)
% Accept a GPU int64[] tag and convert it to a human-readable struct.
% cudaCommon.h enumarates what the indices refer to

partdir = '';
switch(double(t(6)) );
	case 1; partdir = 'X';
	case 2; partdir = 'Y';
	case 3; partdir = 'Z';
end

b = (numel(t) - 8)/2;

x = reshape(t(9:end), [2 b]);

s = struct('arrayDimensions', double(t(1:3)'), ...
           'numel', prod(double(t(1:3))), ...
           'haloSize', double(t(5)), ...
           'exteriorHalos', double(t(8)), ...
           'partitionDirection', partdir, ...
           'numGPUs', double(t(7)), ...
           'slabInfo', double(t(4)), ...
           'pointers', x);

end
