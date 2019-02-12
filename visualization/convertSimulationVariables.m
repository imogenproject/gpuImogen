function convertSimulationVariables()
%
%

SP = SavefilePortal('./');

IC = SP.getInitialConditions();

nranks = IC.ini.geomgr.context.size;

firstmove = 1;

% for every frame
for N = 1:SP.numFrames
    % and every rank,
    for r = 1:nranks
        % load the frame, convert it depending on contents,
        F_i = SP.setFrame(N, r-1);
        fn =  SP.getSegmentFilename(N, r-1);
        
        if isfield(F_i, 'momX')
            Q_i = cvtToPrimitive(F_i);
            if firstmove; disp('Converting to primitive variables.'); firstmove = 0; end
        else
            Q_i = cvtToConservative(F_i);
            if firstmove; disp('Converting to conservative variables.'); firstmove = 0; end
        end
        
        delete(fn);
        
        % and spit it back out.
        if strcmp(fn((end-3):end), '.mat')
            eval([sliceName '= Q_i;']);
            save(fn, sliceName);
        end
        if strcmp(fn((end-2):end),'.nc')
            util_Frame2NCD(fn, Q_i);
        end
        if strcmp(fn((end-2):end), '.h5')
            util_Frame2HDF(fn, Q_i);
        end
    end
end


end

function y = cvtToPrimitive(x)
y = x;

minv = 1./x.mass;

psq = x.momX.*x.momX;
y.velX = x.momX.*minv;
y = rmfield(y, 'momX');

psq = psq + x.momY.*x.momY;
y.velY = x.momY.*minv;
y = rmfield(y, 'momY');

psq = psq + x.momZ.*x.momZ;
y.velZ = x.momZ.*minv;
y = rmfield(y, 'momZ');

y.eint = x.ener - .5*psq.*minv;
y = rmfield(y, 'ener');

if isfield(x, 'mass2')
    minv = 1./x.mass2;
    
    psq = x.momX2.*x.momX2;
    y.velX2 = x.momX2.*minv;
    y = rmfield(y, 'momX2');
    
    psq = psq + x.momY2.*x.momY2;
    y.velY2 = x.momY2.*minv;
    y = rmfield(y, 'momY2');
    
    psq = psq + x.momZ2.*x.momZ;
    y.velZ2 = x.momZ2.*minv;
    y = rmfield(y, 'momZ2');
    
    y.eint2 = x.ener2 - .5*psq.*minv;
    y = rmfield(y, 'ener2');
end

end

function y = cvtToConservative(x)
    y = x;

    vsq = y.velX.*y.velX;
    y.momX = y.velX.*y.mass;
    y = rmfield(y, 'velX');

    vsq = vsq + y.velY.*y.velY;
    y.momY = y.velY.*y.mass;
    y = rmfield(y, 'velY');

    vsq = vsq + y.velZ.*y.velZ;
    y.momZ = y.velZ.*y.mass;
    y = rmfield(y, 'velZ');

    y.ener = y.eint + .5*x.mass.*vsq;
    y = rmfield(y, 'eint');
    
if isfield(x, 'mass2')
    vsq = x.velX2.*x.mvelX2;
    y.momX2 = x.velX2.*x.mass2;
    y = rmfield(y, 'velX2');
    
    vsq = vsq + x.velY2.*x.velY2;
    y.momY2 = x.velY2.*x.mass2;
    y = rmfield(y, 'velY2');
    
    vsq = vsq + x.velZ2.*x.velZ;
    y.momZ2 = x.velZ2.*x.mass2;
    y = rmfield(y, 'velZ2');
    
    y.ener2 = x.eint2 + .5*vsq.*x.mass2;
    y = rmfield(y, 'eint2');
end


end
