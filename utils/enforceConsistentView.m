function timeTaken = enforceConsistentView(directory, timeout, existence)
% enforceConsistentView(directory, timeout, onlyExistence)
% Must be called by all ranks
% Goes to 'dir', and polls #files, filenames & total file size
% Does not return until all ranks agree on these
% If input argument onlyExistence is present, returns once
% FIXME: the existing hashes are terrible

if nargin == 0
    directory = pwd();
end

warnAboutTimeout = 0;
if nargin < 2
    timeout = 600;
    if mpi_amirank0(); warnAboutTimeout = 1; end
end


theSame = 0;
wait = 0.0625;

tic

if nargin >= 3
    % we are only checking that 'directory' exists for everyone
    % Wait forever until save directory is globally visible
    while true
        ready = exist(directory,'dir');
        
        if ready == 7; break; end
        pause(wait);
        wait = wait*2; if(wait > 4); wait = 4; end
    end
    
    fprintf('%i ', int32(mpi_myrank()));
    mpi_barrier();
    
    SaveManager.logPrint('\n');
    
    timeTaken = toc();
else    
    while theSame == 0
        names = [];
        fileList = dir(directory);
        totalBytes = 0;
        
        for N = 1:numel(fileList)
            names = [names fileList(N).name];
            totalBytes = totalBytes + fileList(N).bytes;
        end
        testvec = [N 1*do_xor_hash(names) totalBytes];
        
        most = mpi_max(testvec);
        least= mpi_min(testvec);
        
        theSame = all(most == least);
        
        if theSame == 0
            pause(wait);
            wait = wait*2; if(wait > 4); wait = 4; end
            
            if toc() > timeout
                break;
            end
        end
    end
    
    timeTaken = toc();
    SaveManager.logPrint('All ranks saw consistent size/name/filecount in %s in %f sec\n', directory, timeTaken);
    
end


if mpi_amirank0() && (timeTaken > 0.1) % do not blather about short times.
    if warnAboutTimeout; fprintf('enforceConsistentView: No timeout given, defaulted to 10 wallclock minutes.\n'); end
end

end

function H = do_xor_hash(s)

H = uint32(3129382447);

l = numel(s);
toadd = 4-mod(l,4);
if toadd ~= 4; s((end+1):(end+toadd)) = s(end); end

q = uint32(s(1:4:end) + 256*s(2:4:end) + 65536 * s(3:4:end) + 16777216 *s(4:4:end));

for n = 1:numel(q)
    H = bitxor(H, q(n));
end

end
