function f = paramToFlow(ms, ma, theta, gamma)

f.rho = 1;
f.P   = 1;
f.gamma = gamma;

c_s = sqrt(f.gamma);

f.vx = ms*c_s;
f.vy = tan(theta)*f.vx;
f.vz = 0;

f.bx = f.vx / ma;
f.by = tan(theta)*f.bx;
f.bz = 0;

end
