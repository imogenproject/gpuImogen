function convertSimulationVariables(varFormat)
% convertSimulationVariables(fmt) will convert all savefiles in the current
% directory to either conservative (varFormat = 'conservative') or
% primitive (varFormat = 'primitive') format.

SP = SavefilePortal('./');

IC = SP.getInitialConditions();

nranks = IC.ini.geomgr.context.size;

SP.setVarFormat(varFormat);

fprintf('WARNING WARNING WARNING\n');
fprintf('THIS FUNCTION WILL ALTER SIMULATION DATA FILES ON DISK!!!\n');

x = input('Input 1 to continue or anything else to abort: ', 's');
if strcmp(s, '1') == 0; return; end

% for every frame
for N = 1:SP.numFrames
    % and every rank,
    for r = 1:nranks
        % load the frame, convert it depending on contents,
        F_i = SP.setFrame(N, r-1);
        fn =  SP.getSegmentFilename(N, r-1);
        
        delete(fn);
        
        % and spit it back out.
        if strcmp(fn((end-3):end), '.mat')
            eval([sliceName '= F_i;']);
            save(fn, sliceName);
        end
        if strcmp(fn((end-2):end),'.nc')
            util_Frame2NCD(fn, F_i);
        end
        if strcmp(fn((end-2):end), '.h5')
            util_Frame2HDF(fn, F_i);
        end
    end
end


end
