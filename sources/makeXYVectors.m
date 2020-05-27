function xyvector = makeXYvectors(geometry)
% This function reads the input GeometryManager and creates the vector [x positions; y positions]

[uv, vv, ~] = geometry.ndgridVecs('pos');
xyvector = GPU_Type([ (uv-geometry.frameRotationCenter(1)) (vv-geometry.frameRotationCenter(2)) ], 1);


end
