cd ~/gpuimogen/run/

a = load('run_plist.mat');
b = load('runpar.mat');

fprintf('Run parameters:\n');
fprintf('    Resolution nx = %i\n', int32(b.runpar.res));
fprintf('    # steps       = %i (%.1f million)\n', int32(b.runpar.steps), .1*round(10*b.runpar.steps / 1e6));
fprintf('Runners:\n');
fprintf('    Expecting %i invocations\n', int32(a.runners));
fprintf('Parameter list:\n');
disp(a.plist)

fprintf('================\nIf any of this is wrong, do not execute!\n');
