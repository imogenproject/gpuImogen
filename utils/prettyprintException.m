function prettyprintException(ME, nmax, extendedString)
% prettyprintException(ME, nmax, extendedString)
% > ME: A Matlab MException object
% > nmax: If present, maximum number of stack entries to print
% > extendedString: Displayed in line with the rest of the output if included.
% This function pretty-prints the MException's stack{} lines,

if nargin < 2; nmax = numel(ME.stack); end
if nmax <= 0; nmax = numel(ME.stack); end

[~, hname] = system('hostname');

fprintf('\n========== IMOGEN HAS ENCOUNTERED AN ERROR IN THE INTERPRETER ==================\nRANK %i (HOSTNAME %s) HAS ENCOUNTERED AN EXCEPTION\nIDENTIFIER: %s\nMESSAGE   : %s\n',mpi_myrank(), hname(1:(end-1)), ME.identifier, ME.message);
if nargin >= 3
    fprintf('USER MESG : %s\n',extendedString);
end
fprintf('========== MATLAB STACK BACKTRACE FOLLOWS: =====================================\n');

for n = 1:nmax
    fprintf('%i: %s:%s at %i\n', n-1, ME.stack(n).file, ME.stack(n).name, ME.stack(n).line);
end
if nmax == 0
    disp('No stack: Error occured in interactive mode');
end

disp('================================================================================');

end
