function topology = parallel_topology(context, numDimensions)
% FIXME This needs to take a grid resolution to do its job right...
% FIXME Currently it simply attempts to make the most logically cubical
% grid, which is profoundly WRONG for simulation grids which are not
% themselves cubical!!!!!!

nranks = context.size;
myrank = context.rank;

if numDimensions == 1
    mycoord = [myrank 0 0];
    
    nProc  = [nranks 1 1];
else; if numDimensions == 2
        p = [factor(nranks) 1 1]; % make sure we can pick (2:... and (3:...)
        
        nProc(1) = prod(p(1:2:end));
        nProc(2) = prod(p(2:2:end));
        nProc(3) = 1;
        
        mycoord(1) = floor(myrank / nProc(2));
        r = myrank - mycoord(1) * nProc(2);
        mycoord(2) = r;
        mycoord(3) = 0;
    else; if numDimensions == 3
            p = factor(nranks);
            
            nProc(1) = prod(p(1:3:end));
            nProc(2) = prod(p(2:3:end));
            nProc(3) = prod(p(3:3:end));
            
            mycoord(1) = mod(myrank, nProc(1));
            r = (myrank - mycoord(1))/nProc(1);
            
            mycoord(2) = mod(r, nProc(2));
            r = (r - mycoord(2))/nProc(2);
            mycoord(3) = r;
        end
    end
end

myleft  = mod(mycoord - 1 + nProc, nProc);
myright = mod(mycoord + 1,         nProc);
% Convert these indexes from tuple to rank
% MPI_Cart_rank should do this...
myleft(1) = tupleToRank([myleft(1) mycoord(2) mycoord(3)], nProc);
myleft(2) = tupleToRank([mycoord(1) myleft(2) mycoord(3)], nProc);
myleft(3) = tupleToRank([mycoord(1) mycoord(2) myleft(3)], nProc);

myright(1) = tupleToRank([myright(1) mycoord(2) mycoord(3)], nProc);
myright(2) = tupleToRank([mycoord(1) myright(2) mycoord(3)], nProc);
myright(3) = tupleToRank([mycoord(1) mycoord(2) myright(3)], nProc);

% Build a big friendly structure & return it after MPI builds the directional communicators
topoA = struct('ndim', numDimensions, 'comm', context.comm, 'coord', mycoord, 'neighbor_left', myleft, 'neighbor_right', myright, 'nproc', nProc, 'dimcomm', [-1 -1 -1]);
topology = mpi_createDimcomm(topoA);

end

function r = tupleToRank(t, nproc)
    r = t(1) + nproc(1)*(t(2) + nproc(2)*t(3));
end

