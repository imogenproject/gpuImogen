function enforceConsistentView(directory, timeout)
% Must be called by all ranks
% Goes to 'dir', and polls #files, filenames & total file size
% Does not return until all ranks agree on these
% FIXME: the existing hashes are terrible

if nargin == 0;
    directory = pwd();
end

warnAboutTimeout = 0;
if nargin < 2
    timeout = 600;
    if mpi_amirank0(); warnAboutTimeout = 1; end
end

% List everything this node sees on the F.S.
LIST = dir('./');

names = [];
totalBytes = 0;

theSame = 0;
wait = 0.25;

tic

while theSame == 0
    for N = 1:numel(LIST)
	    names = [names LIST(N).name];
        totalBytes = totalBytes + LIST(N).bytes;
    end

    names = names + 0; % get Ml to cast to numeric

    testvec = [N names totalBytes];

    most = mpi_max(testvec);
    least= mpi_min(testvec);

    theSame = all(most == least);

    if theSame == 0;
        wait = wait*2; if(wait > 8) wait = 8; end
        if wait > .25; pause(wait); end

        if toc() > timeout
	    
	end

    end
end

elapsed = toc();

if mpi_amirank0() && (elapsed > 0.05) % do not blather about short times.
    if warnAboutTimeout; fprintf('enforceConsistentView: No timeout given, defaulted to 10 wallclock minutes.\n'); end
    fprintf('All ranks saw consistent size/name/filecount in %s in %f sec\n', directory, elapsed);
end

end
