function writeSimInitializer(run, IC)

%--- Store everything but Q(x,t0) in a new IC file in the save directory ---%
if run.save.FSAVE
    GIS = GlobalIndexSemantics();

    IC.mass = []; IC.ener   = [];
    IC.mom = [];  IC.magnet = [];
    IC.amResuming = 1;
    IC.originalPathStruct = run.paths.serialize();

    save(sprintf('%s/SimInitializer_rank%i.mat',run.paths.save,GIS.context.rank),'IC');
end


end

