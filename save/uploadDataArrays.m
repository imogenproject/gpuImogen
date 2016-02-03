function [mass ener mom mag DataHolder] = uploadDataArrays(FieldSource, run, statics)

    DataHolder = GPU_Type(FieldSource.mass);
    DataHolder.createSlabs(5); % [rho E px py pz] slabs

    a = GPU_getslab(DataHolder, 0);

    mass = FluidArray(ENUM.SCALAR, ENUM.MASS, a, run, statics);

    a = GPU_setslab(DataHolder, 1, FieldSource.ener);
    ener = FluidArray(ENUM.SCALAR, ENUM.ENER, a, run, statics);

    mom  = FluidArray.empty(3,0);
    mag  = MagnetArray.empty(3,0);
    fieldnames = {'momX','momY','momZ','magX','magY','magZ'};

    for i = 1:3;
        a = GPU_setslab(DataHolder, 1+i, getfield(FieldSource, fieldnames{i}) );
        mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, a, run, statics);
        if run.pureHydro == 0
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, getfield(FieldSource, fieldnames{i+3}), run, statics);
        else
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run, statics);
        end
     end


end
