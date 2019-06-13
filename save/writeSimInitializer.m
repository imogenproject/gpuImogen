function writeSimInitializer(run, IC)

%--- Store everything but Q(x,t0) in a new IC file in the save directory ---%
if run.save.FSAVE
    parallels = ParallelGlobals();

    IC.fluids = [];
    
    IC.magX = [];
    IC.magY = [];
    IC.magZ = [];
    
    IC.amResuming = 1;
    IC.originalPathStruct = run.paths.serialize();

    IC.potentialField.field = [];

    save(sprintf('%s/SimInitializer_rank%i.mat',run.paths.save,parallels.context.rank),'IC');
end


end

