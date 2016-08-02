function result = FluidQty(f, thing)

if nargin < 2; disp('Usage: load frame (util_LoadWholeFrame), then FluidQty(frame, quantity) where quantity can be: pressure (total if B != 0), gaspressure, vx, vy, vz, speed, vorticity, compression, and if B != 0, magpressure, plasmabeta or current'); end

% Some of these are local functions and 'easy' to do,
% Others are differential expressions and require actual work
if strcmpi(thing, 'pressure')
    result = FluidQty(f, 'gaspressure') + FluidQty(f, 'magpressure');
elseif strcmpi(thing, 'gaspressure')
    result = (f.GAMMA-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass - FluidQty(f,'magpressure'));
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
    result.X = d_di(vz, 2, f.DGRID{2}) - d_di(vy, 3, f.DGRID{3});
    result.Y =-d_di(vz, 1, f.DGRID{1}) + d_di(vx, 3, f.DGRID{3});
    result.Z = d_di(vy, 1, f.DGRID{1}) - d_di(vx, 2, f.DGRID{2});
elseif strcmpi(thing, 'compression') % = divergence(V)
    vx = f.momX./f.mass;
    vy = f.momY./f.mass;
    vz = f.momZ./f.mass;
    result = d_di(vx, 1, f.DGRID{1}) + d_di(vy, 2, f.DGRID{2}) + d_di(vz, 3, f.DGRID{3});
elseif strcmpi(thing, 'deldotb') % ought to be ZERO
    result = (circshift(f.magX,[-1 0 0]) - f.magX) ./ f.DGRID{1} + ...
             (circshift(f.magY,[0 -1 0]) - f.magY) ./ f.DGRID{2} + ...
             (circshift(f.magZ,[0 0 -1]) - f.magZ) ./ f.DGRID{3};
elseif strcmpi(thing, 'magpressure') % = B^2 / 2
    result = .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
    result = FluidQty(f,'gaspressure') ./ FluidQty(f,'magpressure');
elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
    result = [];
end

end

function result = d_di(array, i, h)
    delta = [0 0 0]; delta(i) = 1;
    result = (circshift(array,-delta) - circshift(array,delta)) ./ h;
end


