function result = util_DerivedQty(f, thing)

if nargin < 2; disp('Usage: load frame (util_LoadWholeFrame), then util_DerivedQty(frame, quantity) where quantity can be: pressure (total if B is not null), gaspressure, vx, vy, vz, speed, vorticity, compression, and if B is not null magpressure, plasmabeta or current'); end

if strcmpi(thing, 'pressure')
    result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass) + (2-f.gamma)*util_DerivedQty(f,'magpressure');
elseif strcmpi(thing, 'gaspressure')
    result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass - .5*(f.magX.^2+f.magY.^2+f.magZ.^2));
elseif strcmpi(thing, 'vx')
    result = f.momX ./ f.mass;
elseif strcmpi(thing, 'vy')
    result = f.momY ./ f.mass;
elseif strcmpi(thing, 'vz')
    result = f.momZ ./ f.mass;
elseif strcmpi(thing, 'speed') % = |mom| / mass
    result = sqrt(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass;
elseif strcmpi(thing, 'vorticity') % = curl(V)
    vx = f.momX./f.mass;
    vy = f.momY./f.mass;
    vz = f.momZ./f.mass;
    result.X = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
    result.Y =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
    result.Z = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2}); 
elseif strcmpi(thing, 'compression') % = del . V
    vx = f.momX./f.mass;
    vy = f.momY./f.mass;
    vz = f.momZ./f.mass;
    result = +d_di(vx, 1, -f.dGrid{1}) + d_di(vy, 2, -f.dGrid{2}) + d_di(vz, 3, -f.dGrid{3});
elseif strcmpi(thing, 'magpressure') % = B^2 / 2
    result = .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
    result = util_DerivedQty(f,'gaspressure') ./ util_DerivedQty(f,'magpressure');
elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
    result.X = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
    result.Y =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
    result.Z = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
end

end

function result = d_di(array, i, h)
    delta = [0 0 0]; delta(i) = 1;
    result = (circshift(array,-delta) - circshift(array,delta)) ./ h;
end
