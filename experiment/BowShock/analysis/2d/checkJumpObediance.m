function checkJumpObediance(imogenFrame, dl, h)

rho = imogenFrame.mass;
vx  = imogenFrame.momX ./ rho;
vy  = imogenFrame.momY ./ rho;
csq = (5/3 - 1)*(5/3)*(imogenFrame.ener./rho - .5*(vx.^2+vy.^2)); % csq = gamma P / rho
bx  = 0;
by  = 0;

%imagesc(csq)
%drho_dx = diff(rho, 1, 1); % differentiate vx
%drho_dy = diff(rho, 1, 2);
shockMeasure = diff(vx,1,1);

p = input('Coordinates of first point near shock: ');
Rsh(:,1) = refineShockCoord(p, abs(shockMeasure), [1 0], 5*h);
p = input('Coordinates of second point near shock: ');
Rsh(:,2) = refineShockCoord(p, abs(shockMeasure), [1 0], 5*h);

NVsh(:,1) = Rsh(:,2) - Rsh(:,1);
NVsh = NVsh/norm(NVsh);

Rsh(:,2) = Rsh(:,1) + dl*NVsh;
Rsh(:,2) = refineShockCoord(Rsh(:,2)', abs(shockMeasure), [1 0], 5*h);

%for (a while) | (we loop back very near our original position) | (we encounter a topological oddity like a branch)
for k = 2:10
	Rsh(:,k) = Rsh(:,k-1) + dl*NVsh(:,k-1);
	Rsh(:,k) = refineShockCoord(Rsh(:,k)', abs(shockMeasure), cross(NVsh(:,k-1), [0 0 1]), 5*h);

	NVsh(:,k) = Rsh(:,k) - Rsh(:,k-1);
	NVsh(:,k) = NVsh(:,k) / norm(NVsh(:,k));

end

usv = cross(NVsh, [0;0;1]*ones([1 size(NVsh,2)]));

VatSH = zeros(size(usv));
VatSH(1,:) = interp2(vx, Rsh(2,:), Rsh(1,:),'linear');
VatSH(2,:) = interp2(vy, Rsh(2,:), Rsh(1,:),'linear');
VatSH(3,:) = 0;

usv = -usv .* (ones([3 1])*sign(dot(usv, VatSH,1)) ) % up stream vector now points upstream

%extrapolate uppts   = Q(P_i + usv_i * h)
%extrapolate downpts = Q(P_i - usv_i * h) for a spacing h of 3-5 cells

%downpred = jumpSolver(uppts)

%enorm = (downpred - downpts) ./ downpred; Get the error in our predicted downstream flow behavior

end

