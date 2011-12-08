function new = refineShockCoord(old, lap, perp, searchlen)

old(3)=0;

perp(3) = 0;
perp = perp/norm(perp);

dh = ((-10:10)'*searchlen/10) * perp + ones([21 1])*old;

vals = interp2(lap, dh(:,2), dh(:,1),'linear');

[dump ind] = max(vals);

new = old(1:2) + ((ind-11)*searchlen/10) * perp(1:2);
new(3)=0;

end
