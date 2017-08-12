function [V err] = input2vector(user, N, padvalue, makeint)
% input2vector(user, N, padvalue, makeint) is a one-stop shop
% covering many common cases of getting an N-vector
%> user: user's input
%> N: number of elments in return vector
%> padvalue: V(++numel(user):N) is padded with this
%> makeint: if nonzero, round()s result
%< V: return vector
%< err: Any 

V = user;

% Make sure we have N elements
V((numel(user)+1):N) = padvalue(1);

% Crop to N if too many
if numel(V) > N; V = V(1:N); end

if makeint; V = round(V); end

V = reshape(V, [numel(V) 1]);

end
