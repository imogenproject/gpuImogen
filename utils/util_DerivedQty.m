function result = util_DerivedQty(f, thing, component)
% util_DerivedQty(f, thing, component) computes all the (simple) derived quantities commonly of interest from the
% conserved variables that Imogen computes:
%>> f: The data frame returned by util_LoadWholeFrame or similar
%>> thing: One of the strings mass, ener, mom, mag, pressure, gaspressure, velocity, speed, vorticity, compression, magpressure, plasmabeta, or current.
%>> component: Required only if thing is a vector, either 0 (returns structure.X .Y .Z vector) or 1/2/3 for X/Y/Z component only

if strcmpi(thing, 'mass') % The 8 conserved variables we have anyway for completeness
    result = f.mass;
elseif strcmpi(thing, 'ener')
    result = f.ener;
elseif strcmpi(thing, 'mom')
    if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mom'', component'); end
    switch component
	case 0; result.X = f.momX; result.Y = f.momY; result.Z = f.momZ;
	case 1; result = f.momX;
	case 2; result = f.momY;
	case 3; result = f.momZ;
    end
elseif strcmpi(thing, 'mag');
    if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mag'', component'); end
    switch component
	case 0; result.X = f.magX; result.Y = f.magY; result.Z = f.magZ;
	case 1; result = f.magX;
	case 2; result = f.magY;
	case 3; result = f.magZ;
    end
elseif strcmpi(thing, 'pressure') % And now derived and/or primitives
    result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass) + (2-f.gamma)*util_DerivedQty(f,'magpressure');
elseif strcmpi(thing, 'gaspressure')
    result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass - .5*(f.magX.^2+f.magY.^2+f.magZ.^2));
elseif strcmpi(thing, 'velocity')
    if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''velocity'', component'); end
    switch component
	case 0; minv = 1./f.mass; result.X = f.momX.*minv; result.Y = f.momY.*minv; result.Z = f.momZ.*minv;
	case 1; result = f.momX./f.mass;
	case 2; result = f.momY./f.mass;
	case 3; result = f.momZ./f.mass;
    end
elseif strcmpi(thing, 'speed') % = |mom| / mass
    result = sqrt(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass;
elseif strcmpi(thing, 'soundspeed') % = sqrt(gamma P / rho)
    P = util_DerivedQty(f, 'gaspressure');
    result = sqrt(f.gamma*P./f.mass);    
elseif strcmpi(thing, 'vorticity') % = curl(V)
    if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''vorticity'', component'); end
    minv = 1./f.mass;
    vx = f.momX.*minv;
    vy = f.momY.*minv;
    vz = f.momZ.*minv;
    clear minv;
    switch component
        case 0; 
	    result.X = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
	    result.Y =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
	    result.Z = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2}); 
        case 1; result = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
        case 2; result =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
        case 3; result = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2}); 
    end
elseif strcmpi(thing, 'compression') % = div(V)
    minv = 1./f.mass;
    vx = f.momX.*minv;
    vy = f.momY.*minv;
    vz = f.momZ.*minv;
    clear minv;
    result = +d_di(vx, 1, -f.dGrid{1}) + d_di(vy, 2, -f.dGrid{2}) + d_di(vz, 3, -f.dGrid{3});
elseif strcmpi(thing, 'magpressure') % = B.B / 2
    result = .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
    result = util_DerivedQty(f,'gaspressure') ./ util_DerivedQty(f,'magpressure');
elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
    if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''current'', component'); end
    switch component
        case 0;
            result.X = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
            result.Y =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
            result.Z = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
        case 1; result = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
        case 2; result =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
        case 3; result = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
    end
end

end

function result = d_di(array, i, h)
    delta = [0 0 0]; delta(i) = 1;
    result = (circshift(array,-delta) - circshift(array,delta)) ./ h;
end
