function writeSimInitializer(run, IC)

%--- Store everything but Q(x,t0) in a new IC file in the save directory ---%
if run.save.FSAVE
    parallels = ParallelGlobals();

    for N = 1:numel(IC.fluids)
        IC.fluids(N).mass = [];
        IC.fluids(N).momX = [];
        IC.fluids(N).momY = [];
        IC.fluids(N).momZ = [];
        IC.fluids(N).ener = [];
    end
    
    IC.magX = [];
    IC.magY = [];
    IC.magZ = [];
    
    IC.amResuming = 1;
    IC.originalPathStruct = run.paths.serialize();

    IC.potentialField.field = [];

    save(sprintf('%s/SimInitializer_rank%i.mat',run.paths.save,parallels.context.rank),'IC');
end


end

