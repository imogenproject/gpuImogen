function L = selectGPUs(gpus_arg)
% Accepts list of GPUs given to imogen at invocation
% This can take three forms:
% (1) A vector of the form [vector, of, CUDA, device #s],
% (2) The scalar magic value -1 works with SLURM and uses
%     getenv('GPU_DEVICE_ORDINAL') to retreive the above vector,
% (3) a file containing fields of the form
%     s(x) = struct('hostname','nodeX','devlist',[gpus_for_first_rank_on_node, for_2nd_rank; ...]);
% If successful, returns the set of GPUs that the rank should initialize in Imogen.

if isa(gpus_arg,'double')
    if gpus_arg(1) == -1
        if mpi_amirank0(); disp('SLURM GPU assignment: all ranks will getenv GPU_DEVICE_ORDINAL'); end
        L = str2num(getenv('GPU_DEVICE_ORDINAL'));
    else
        L = gpus_arg;
    end

    return;
end

failure = 0;
bi = mpi_basicinfo();


try
    load(gpus_arg);
catch
    failure = 1;
    fprintf('Rank %i tried to open %s and failed: Run failing.\n', int32(bi(2)), gpus_arg);
    mpi_errortest(failure);
end

R = ranksOnHost();

[dump, myself] = system('hostname');
myself = deblank(myself);


kmax = numel(gpuList);
k = 1;
while (k <= kmax) && (strcmp(myself, gpuList(k).hostname) == 0)
    k = k + 1;
end

if k > kmax % no entry for ue? Is there a default?
    fprintf('Rank %i: No entry in %s naming my host (%s); Trying "default"...\n', int32(bi(2)), gpus_arg, myself);
    if strcmp(gpuList(1).hostname, 'default')
        devs = gpuList(1);
    else
        failure = 1;
        fprintf('Rank %i: No default entry in host/device enumeration file; Failing...\n', int32(bi(2)));
        mpi_errortest(failure);
    end
else
    devs = gpuList(k);
end

if size(devs.devlist,1) >= R(1) % If the device list is usable,
    L = devs(1).devlist(R(2)+1,:);
else
    failure = 1;
end

mpi_errortest(failure);

end
