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

theSame = 0;
wait = 0.25;

tic

mr = mpi_myrank();

while theSame == 0
    names = [];
    fileList = dir(directory);
    totalBytes = 0;

    for N = 1:numel(fileList)
	    names = [names fileList(N).name];
        totalBytes = totalBytes + fileList(N).bytes;
    end
    testvec = [N 1*do_xor_hash(names) totalBytes]

    most = mpi_max(testvec);
    least= mpi_min(testvec);

    theSame = all(most == least);

    if theSame == 0;
        wait = wait*2; if(wait > 8) wait = 8; end
        if wait > .25; pause(wait); end

        if toc() > timeout
	    break;
	end
    end
end

elapsed = toc();

if mpi_amirank0() && (elapsed > 0.05) % do not blather about short times.
    if warnAboutTimeout; fprintf('enforceConsistentView: No timeout given, defaulted to 10 wallclock minutes.\n'); end
    fprintf('All ranks saw consistent size/name/filecount in %s in %f sec\n', directory, elapsed);
end

end

function H = do_xor_hash(s)

H = uint32(3129382447);

l = numel(s);
toadd = 4-mod(l,4);
if toadd ~= 4; s((end+1):(end+toadd)) = s(end); end

q = uint32(s(1:4:end) + 256*s(2:4:end) + 65536 * s(3:4:end) + 16777216 *s(4:4:end));

for n = 1:numel(q);
    H = bitxor(H, q(n));
end

end
