function result = indexSet_fromLogical(logic)
% Given the dimensions in dims and sets of x, y and z coordinates,
% This function takes the volume spanned by the cartesian of xset x yset x zset
% and returns [linear x y z] Nx4 matrix corresponding to the linear and x y z
% indicies corresponding to the selected points
% Logical Awesome: My Codes Are Perfect

[u, v, w] = ndgrid(1:size(logic,1), 1:size(logic,2), 1:size(logic,3) );

linears = find(logic);
result = [linears u(linears) v(linears) w(linears)];

end 
