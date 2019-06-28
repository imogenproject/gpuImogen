function Y = fillICFromFrame(IC, frame)

Y = IC;

if isempty(IC.fluids)  % for old runs that dumped the whole struct 
    if isfield(frame, 'mass2') % two fluids
        Y.fluids = struct('mass',frame.mass, ...
            'momX', frame.momX, 'momY', frame.momY, 'momZ', frame.momZ, ...
            'ener', frame.ener, 'details', [], 'bcData', []);
        
        Y.fluids(2) = struct('mass',frame.mass2, ...
            'momX', frame.momX2, 'momY', frame.momY2, 'momZ', frame.momZ2, ...
            'ener', frame.ener2, 'details', [], 'bcData', []);
    else % one fluid
        Y.fluids = struct('mass',frame.mass, ...
            'momX', frame.momX, 'momY', frame.momY, 'momZ', frame.momZ, ...
            'ener', frame.ener, 'details', [], 'bcData', []);
    end
else
    Y.fluids(1).mass = frame.mass;
    Y.fluids(1).momX = frame.momX;
    Y.fluids(1).momY = frame.momY;
    Y.fluids(1).momZ = frame.momZ;
    Y.fluids(1).ener = frame.ener;
    if isfield(frame, 'mass2') % two fluids
        Y.fluids(2).mass = frame.mass2;
        Y.fluids(2).momX = frame.momX2;
        Y.fluids(2).momY = frame.momY2;
        Y.fluids(2).momZ = frame.momZ2;
        Y.fluids(2).ener = frame.ener2;
    end
end

end