function C = stokesDragCoeff(Re)
% C = stokesDragCoeff(Re) gives the drag coefficient C from F = C(V)*V (up to a certain
% normalization)
% C = { Re < 1         | 12/Re
%       1 < Re < 784.5 | 12 Re^-.6
%      784.5 < Re      | 0.22
%if Re < 905.2897741756132
%    C = 12 ./ Re + 2*Re.^-.33333333333333;
%else
%    C = 0.22;
%end

C = zeros(size(Re));
C(Re < 1) = 12 ./ Re(Re < 1);

C(Re > 784.5084) = .22;

q = (Re > 1) & (Re < 784.5084);
C(q) = 12*Re(q).^-.6;

end