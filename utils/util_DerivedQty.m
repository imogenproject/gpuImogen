function result = util_DerivedQty(f, thing)

if nargin < 2; disp('Usage: load frame (util_LoadWholeFrame), then util_DerivedQty(frame, quantity) where quantity can be: pressure (total if B is not null), gaspressure, vx, vy, vz, speed, vorticity, compression, and if B is not null magpressure, plasmabeta or current'); end

if strcmpi(thing, 'pressure')
    result = f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass;
elseif strcmpi(thing, 'gaspressure')
    result = f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass - .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
elseif strcmpi(thing, 'vx')
    result = f.momX ./ f.mass;
elseif strcmpi(thing, 'vy')
    result = [];
elseif strcmpi(thing, 'speed') % = |mom| / mass
    result = [];
elseif strcmpi(thing, 'vorticity') % = curl(V)
    result = [];
elseif strcmpi(thing, 'compression') % = del . V
    result = [];
elseif strcmpi(thing, 'magpressure') % = B^2 / 2
    result = [];
elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
    result = []
elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
    result = []
end

end
