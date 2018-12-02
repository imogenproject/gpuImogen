function runTestExchange(asize)

myrank = -1;
% this is copypasta from run/starterRun.m...
if ~mpi_isinitialized()
        % Start up PGW\
        % defaults to a 3D topology
        context = parallel_start();
        myrank = context.rank;
        topology = parallel_topology(context, 3);

        % This will be init'd & stores itself as a persistent var
        parInfo = ParallelGlobals(context, topology); %#ok<NASGU>
     
        if context.rank==0
            fprintf('\n---------- MPI Startup\nFirst start: MPI is now ready. Roll call:\n'); end
        mpi_barrier();
        fprintf('Rank %i ready.\n', int32(context.rank));
        mpi_barrier();
    else
        if context.rank == 0
            fprintf('Well, mpi is already on. not expecting that.\n');
        end
    end

% create some garbage and exchange halos

for ohLawdy = 1:3
    myarray = rand(asize) + myrank; 
predat = myarray([1:8 (end-7):end],1,1)';
%myarray'
    generateTestExchanges(myarray, topology);
fprintf('RANK %i: one data row before & after xchg: ', int32(context.rank)); 
disp([predat; myarray([1:8 (end-7):end],1,1)']);
%myarray'
end



end
