function result = util_DerivedQty(f, thing, component, fluidno)
% util_DerivedQty(f, thing, component) computes all the (simple) derived quantities commonly of
% interest from the conserved variables that Imogen computes:
%>> f: The data frame as returned by util_LoadWholeFrame or util_LoadFrameSegment
%>> thing: One of the strings mass, ener, mom, mag, pressure, gaspressure, velocity, speed, vorticity, compression, magpressure, plasmabeta, or current.
%>> component: Required only if thing is a vector; Then either 0 (returns structure.X .Y .Z
%       vector) or 1/2/3 for X/Y/Z component only
%>> fluidno: For cases where this is meaningful, specifies to return either the fluid 1 (gas) or
%     fluid 2 (dust) quantity; Defaults to gas

if nargin < 4; fluidno = 1; end

% FIXME: this is a lame hack and really the ensight exporter needs to be fixed instead
if thing(end) == '2'
    fluidno = 2;
    thing = thing(1:(end-1));
end

if isfield(f, 'momX') % if variables are in conservative form
    if strcmpi(thing, 'mass') % The 8 conserved variables we have anyway for completeness
        if fluidno == 1; result = f.mass; else; result = f.mass2; end
    elseif strcmpi(thing, 'ener')
        if fluidno == 1; result = f.ener; else; result = f.ener2; end
    elseif strcmpi(thing, 'mom')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mom'', component'); end
        switch component
            case 0; if fluidno == 1; result.X = f.momX; result.Y = f.momY; result.Z = f.momZ; else; result.X = f.momX2; result.Y = f.momY2; result.Z = f.momZ2; end
            case 1; if fluidno == 1; result = f.momX; else; result = f.momX2; end
            case 2; if fluidno == 1; result = f.momY; else; result = f.momY2; end
            case 3; if fluidno == 1; result = f.momZ; else; result = f.momZ2; end
        end
    elseif strcmpi(thing, 'mag')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mag'', component'); end
        switch component
            case 0; result.X = f.magX; result.Y = f.magY; result.Z = f.magZ;
            case 1; result = f.magX;
            case 2; result = f.magY;
            case 3; result = f.magZ;
        end
        % And now derived and/or primitives
    elseif strcmpi(thing, 'pressure')
        result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass) + (2-f.gamma)*util_DerivedQty(f,'magpressure');
    elseif strcmpi(thing, 'gaspressure')
        result = (f.gamma-1)*(f.ener - .5*(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass - .5*(f.magX.^2+f.magY.^2+f.magZ.^2));
    elseif strcmpi(thing, 'velocity')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''velocity'', component'); end
        switch component
            case 0
                if fluidno == 1
                    minv = 1./f.mass; result.X = f.momX.*minv; result.Y = f.momY.*minv; result.Z = f.momZ.*minv;
                else
                    minv = 1./f.mass2; result.X = f.momX2.*minv; result.Y = f.momY2 .* minv; result.Z = f.momZ2 .* minv;
                end
            case 1; if fluidno == 1; result = f.momX./f.mass; else; result = f.momX2 ./ f.mass2; end
            case 2; if fluidno == 1; result = f.momY./f.mass; else; result = f.momY2 ./ f.mass2; end
            case 3; if fluidno == 1; result = f.momZ./f.mass; else; result = f.momZ2 ./ f.mass2; end
        end
    elseif strcmpi(thing, 'speed') % = |mom| / mass
        if fluidno == 1; result = sqrt(f.momX.^2+f.momY.^2+f.momZ.^2)./f.mass; else; result = sqrt(f.momX2.^2 + f.momY2.^2 + f.momZ2.^2)./f.mass; end
    elseif strcmpi(thing, 'soundspeed') % = sqrt(gamma P / rho)
        P = util_DerivedQty(f, 'gaspressure');
        result = sqrt(f.gamma*P./f.mass);
    elseif strcmpi(thing, 'vorticity') % = curl(V)
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''vorticity'', component'); end
        if fluidno == 1
            minv = 1./f.mass;
            vx = f.momX.*minv;
            vy = f.momY.*minv;
            vz = f.momZ.*minv;
        else
            minv = 1./f.mass2;
            vx = f.momX2.*minv;
            vy = f.momY2.*minv;
            vz = f.momZ2.*minv;
        end
        clear minv;
        switch component
            case 0
                result.X = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
                result.Y =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
                result.Z = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2});
            case 1; result = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
            case 2; result =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
            case 3; result = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2});
        end
    elseif strcmpi(thing, 'compression') % = div(V)
        if fluidno == 1
            minv = 1./f.mass;
            vx = f.momX.*minv;
            vy = f.momY.*minv;
            vz = f.momZ.*minv;
        else
            minv = 1./f.mass2;
            vx = f.momX2.*minv;
            vy = f.momY2.*minv;
            vz = f.momZ2.*minv;
        end
        
        clear minv;
        result = d_di(vx, 1, -f.dGrid{1}) + d_di(vy, 2, -f.dGrid{2}) + d_di(vz, 3, -f.dGrid{3});
    elseif strcmpi(thing, 'magpressure') % = B.B / 2
        result = .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
    elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
        result = util_DerivedQty(f,'gaspressure') ./ util_DerivedQty(f,'magpressure');
    elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''current'', component'); end
        switch component
            case 0
                result.X = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
                result.Y =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
                result.Z = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
            case 1; result = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
            case 2; result =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
            case 3; result = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
        end
    elseif strcmpi(thing, '2fluid_dv')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''velocity'', component'); end
        switch component
            case 0
                minv = 1./f.mass; minv2 = 1./f.mass2;
                result.X = f.momX.*minv - f.momX2.*minv2;
                result.Y = f.momY.*minv - f.momY2.*minv2;
                result.Z = f.momZ.*minv - f.momZ2.*minv2;
            case 1; result = f.momX ./ f.mass - f.momX2 ./ f.mass2;
            case 2; result = f.momY ./ f.mass - f.momY2 ./ f.mass2;
            case 3; result = f.momZ ./ f.mass - f.momZ2 ./ f.mass2;
        end
    end
else % if variables are in primitive form
    if strcmpi(thing, 'mass') % The 8 conserved variables we have anyway for completeness
        if fluidno == 1; result = f.mass; else; result = f.mass2; end
    elseif strcmpi(thing, 'ener')
        if fluidno == 1
            result = f.eint + .5*f.mass.*(f.velX.^2+f.velY.^2+f.velZ.^2);
        else
            result = f.eint2 + .5*f.mass2.*(f.velX2.^2+f.velY2.^2+f.velZ2.^2);
        end
    elseif strcmpi(thing, 'mom')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mom'', component'); end
        switch component
            case 0
                if fluidno == 1
                    result.X = f.mass.*f.velX; result.Y = f.mass.*f.velY; result.Z = f.mass.*f.velZ;
                else
                    result.X = f.mass2.*f.velX2; result.Y = f.mass2.*f.velY2; result.Z = f.mass2.*f.velZ2;
                end
            case 1; if fluidno == 1; result = f.mass.*f.velX; else; result = f.mass2.*f.velX2; end
            case 2; if fluidno == 1; result = f.mass.*f.velY; else; result = f.mass2.*f.velY2; end
            case 3; if fluidno == 1; result = f.mass.*f.velZ; else; result = f.mass2.*f.velZ2; end
        end
    elseif strcmpi(thing, 'mag')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''mag'', component'); end
        switch component
            case 0; result.X = f.magX; result.Y = f.magY; result.Z = f.magZ;
            case 1; result = f.magX;
            case 2; result = f.magY;
            case 3; result = f.magZ;
        end
        % And now derived and/or primitives
    elseif strcmpi(thing, 'pressure')
        result = (f.gamma-1)*f.eint + util_DerivedQty(f,'magpressure');
    elseif strcmpi(thing, 'gaspressure')
        result = (f.gamma-1)*f.eint;
    elseif strcmpi(thing, 'velocity')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''velocity'', component'); end
        switch component
            case 0
                if fluidno == 1
                    result.X = f.velX; result.Y = f.velY; result.Z = f.velZ;
                else
                    result.X = f.velX2; result.Y = f.velY2; result.z = f.velZ2;
                end
            case 1; if fluidno == 1; result = f.velX; else; result = f.velX2; end
            case 2; if fluidno == 1; result = f.velY; else; result = f.velY2; end
            case 3; if fluidno == 1; result = f.velZ; else; result = f.velZ2; end
        end
    elseif strcmpi(thing, 'speed') % = |mom| / mass
        if fluidno == 1; result = sqrt(f.velX.^2+f.velY.^2+f.velZ.^2); else; result = sqrt(f.velX2.^2 + f.velY2.^2 + f.velZ2.^2); end
    elseif strcmpi(thing, 'soundspeed') % = sqrt(gamma P / rho)
        P = util_DerivedQty(f, 'gaspressure');
        result = sqrt(f.gamma*P./f.mass);
        % FIXME HELP BUG vorticity calculation obviously does not work as written below in
        % cylindrical coordinates!
    elseif strcmpi(thing, 'vorticity') % = curl(V)
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''vorticity'', component'); end
        if fluidno == 1
            vx = f.velX;
            vy = f.velY;
            vz = f.velZ;
        else
            vx = f.velX2;
            vy = f.velY2;
            vz = f.velZ2;
        end
        switch component
            case 0
                result.X = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
                result.Y =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
                result.Z = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2});
            case 1; result = d_di(vz, 2, f.dGrid{2}) - d_di(vy, 3, f.dGrid{3});
            case 2; result =-d_di(vz, 1, f.dGrid{1}) + d_di(vx, 3, f.dGrid{3});
            case 3; result = d_di(vy, 1, f.dGrid{1}) - d_di(vx, 2, f.dGrid{2});
        end
    elseif strcmpi(thing, 'compression') % = div(V)
        if fluidno == 1
            vx = f.velX;
            vy = f.velY;
            vz = f.velZ;
        else
            vx = f.velX2;
            vy = f.velY2;
            vz = f.velZ2;
        end
        result = d_di(vx, 1, -f.dGrid{1}) + d_di(vy, 2, -f.dGrid{2}) + d_di(vz, 3, -f.dGrid{3});
    elseif strcmpi(thing, 'magpressure') % = B.B / 2
        result = .5*(f.magX.^2+f.magY.^2+f.magZ.^2);
    elseif strcmpi(thing, 'plasmabeta') % = pgas / pmag
        result = util_DerivedQty(f,'gaspressure') ./ util_DerivedQty(f,'magpressure');
    elseif strcmpi(thing, 'current') % = curl(B) as we neglect displacement current
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''current'', component'); end
        switch component
            case 0
                result.X = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
                result.Y =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
                result.Z = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
            case 1; result = d_di(f.magZ, 2, f.dGrid{2}) - d_di(f.magY, 3, f.dGrid{3});
            case 2; result =-d_di(f.magZ, 1, f.dGrid{1}) + d_di(f.magX, 3, f.dGrid{3});
            case 3; result = d_di(f.magY, 1, f.dGrid{1}) - d_di(f.magX, 2, f.dGrid{2});
        end
    elseif strcmpi(thing, '2fluid_dv')
        if nargin < 3; error('component (0 = all, 1/2/3 = vector part) required: util_DerivedQty(f, ''velocity'', component'); end
        switch component
            case 0
                result.X = f.velX - f.velX2;
                result.Y = f.velY - f.velY2;
                result.Z = f.velZ - f.velZ2;
            case 1; result = f.velX - f.velX2;
            case 2; result = f.velY - f.velY2;
            case 3; result = f.velZ - f.velZ2;
        end
    end
end % handled both variable representations!

end

function result = d_di(array, i, h)
    delta = [0 0 0]; delta(i) = 1;
    result = (circshift(array,-delta) - circshift(array,delta)) ./ h;
end
