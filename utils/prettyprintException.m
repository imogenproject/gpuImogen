function prettyprintException(ME, nmax)
% > ME: A Matlab MException object
% This function pretty-prints the MException's stack{} lines,

if nargin < 2; nmax = numel(ME.stack); end

fprintf('\n================================================================================\nRANK %i HAS ENCOUNTERED AN EXCEPTION\nIDENTIFIER: %s\nMESSAGE   : %s\n=========================== STACK BACKTRACE FOLLOWS ============================\n', mpi_myrank(), ME.identifier, ME.message);

for n = 1:nmax;
    fprintf('%i: %s:%s at %i\n', n-1, ME.stack(n).file, ME.stack(n).name, ME.stack(n).line);
end
disp('================================================================================');

end
