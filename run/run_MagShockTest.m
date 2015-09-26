
% Runs a magnetic shock tube problem (Brio-Wu tube)

%--- Initialize test ---%
grid = [1024 16 1];
run               = MagneticShockTubeInitializer(grid);

run.alias         = '';
run.info          = 'Magnetic shock tube test';
run.notes         = '';
run.iterMax = 1000;

run.bcMode.x = 'const';
run.bcMode.y = 'circ';
run.bcMode.z = 'circ';

%--- Run tests ---%
if (true) % Primary test
    run.alias  = [run.alias, '_BXBY'];
    run.xField = true;
    run.yField = true;
    run.zField = false;
    run.direction = MagneticShockTubeInitializer.X;

    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

if (false) % Secondary test
    run.alias  = [run.alias, '_BXBZ'];
    run.xField = true;
    run.yField = false;
    run.zField = true;
    run.direction = MagneticShockTubeInitializer.X;

    IC = run.saveInitialCondsToStructure();
    outdir = imogen(IC);
end

