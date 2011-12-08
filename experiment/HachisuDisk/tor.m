function xinv = tor(r, th, z, rp, thp, zp)

xi = (r^2 + rp^2 + (z-zp)^2)/(2*r*rp);

xinv = 0;
for m = -10:10
    
[k e] = ellipke(sqrt(2/(xi + 1)));
    xinv = xinv + exp(1i*m*(thp-th)) * (-1/sqrt(2*xi - 2)) * e;
end

xinv = xinv / (pi*sqrt(r*rp));

end
