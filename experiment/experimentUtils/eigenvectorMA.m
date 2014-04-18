function [ev omega] = eigenvectorMA(rho, csq, v, b, k, wavetype)
% This function returns the fast MA eigenvector coefficients [1 dvx dvy dvz dbx dby dbz]
% associated with the plane wave posessing the given real k.
% wavetype = [-2: fast bkwd, -1: slow bkwd, 1: slow fwd, 2: fast fwd]
% where forward == re[w] > 0 and backwards == re[w] < 0
% for exp[i(k.r - wt)]

v=reshape(v,[3 1]);
b=reshape(b,[3 1]);
k=reshape(k,[1 3]);

% Get into the DeHoffman-Teller frame s.t. v cross B = 0
E_dht = cross(v, b);
V_dht = cross(E_dht, b);

v = v - V_dht;

% Rotate into the plane such that vz/bz vanish

zangle = atan2(b(3),b(2));

R = [1 0 0; 0 cos(zangle) sin(zangle); 0 -sin(zangle) cos(zangle)];
v=R*v;
b=R*b;
k=k*R; % such that k.r remains invariant as it must

% Solve the dispersion relation
bqA = rho;
bqB = -k*k' * (b'*b + csq * rho);
bqC = csq*(k*k')*(k*b)^2;

% Select lambda such that dispersion relation is satisfied
% Alternative: solveQuartic(bqA, 0, bqB, 0, bqC) gives same roots
if abs(wavetype) == 2
    lambda = -sign(wavetype)*sqrt((-bqB + sqrt(bqB*bqB - 4*bqA*bqC))/(2*bqA));
else
    lambda = -sign(wavetype)*sqrt((-bqB - sqrt(bqB*bqB - 4*bqA*bqC))/(2*bqA));
end

% Evalute the vector
kdb = k*b;
ksq = k*k';
bsq = b'*b;

ev(1) = 1;
ev(2) = (csq*(b(1)*kdb*ksq/(lambda*rho) - k(1)*lambda))/(lambda^2*rho-bsq*ksq);
ev(3) = (csq*(b(2)*kdb*ksq/(lambda*rho) - k(2)*lambda))/(lambda^2*rho-bsq*ksq);
ev(4) = (csq*                             k(3)*lambda) /(bsq*ksq - rho*lambda^2);
ev(5) = (csq*(k(1)*kdb - b(1)*ksq))/(bsq*ksq - rho*lambda^2);
ev(6) = (csq*(k(2)*kdb - b(2)*ksq))/(bsq*ksq - rho*lambda^2);
ev(7) = (csq* k(3)*kdb)            /(bsq*ksq - rho*lambda^2);

omega = k*v - lambda;
end
