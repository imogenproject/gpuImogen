function L = selectGPUs(gpus_arg)
% Accepts list of GPUs given to imogen at invocation
% This is either of the form [vector of CUDA device #2]
% or a file containing fields of the form
% s(x) = struct('hostname','nodeX','devlist',[gpus_for_first_rank_on_node, for_2nd_rank; ...]);
% If successful, returns the set of GPUs that the rank should initialize in Imogen.

if isa(gpus_arg,'double')
    L = gpus_arg;
end

failure = 0;

try
    load(gpus_arg)
catch
    failure = 1;
end

R = ranksOnHost();

[dump, myself] = system('hostname');
myself = deblank(myself);


kmax = numel(gpuList);
k = 1;
while (k <= kmax) && (strcmp(myself, gpuList(k).hostname) == 0)
gpuList(k).hostname
myself
    k = k + 1
end


if k > kmax % no entry for ue? Is there a default?
    if strcmp(gpuList(1).hostname, 'default')
        devs = gpuList(1);
    else
        failure = 1;
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
