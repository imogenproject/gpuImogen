function writeEnsightMasterFiles(basename, range, SP, varset, timeNormalization, reverseIndexOrder)

frame = SP.jumpToLastFrame();

CASE = fopen([basename '.case'], 'w');

% prepare format info
fprintf(CASE, 'FORMAT\ntype: ensight gold\n');

% prepare geometry info
fprintf(CASE, '\nGEOMETRY\n');
fprintf(CASE, 'model: 1 %s.geom\n', basename);
fprintf('writing geometry file\n');
% it needs the savefile portal to access the initializer & build a geometry manager in case of cylindrical coordinates
makeEnsightGeometryFile(SP, frame, basename, reverseIndexOrder);

% prepare variables
fprintf(CASE, '\nVARIABLE\n');

% fixme: There should be a way to interrogate derived quantity tensor types w/o computing them fully...

for n = 1:numel(varset)
    % vector quantities are structures with fields S.[X, Y, Z]
    isVector = isa(frame.(varset{n}), 'struct');

    if isVector
        fprintf(CASE, 'vector per node: 1 %s %s.%s.****\n', varset{n}, basename, varset{n});
    else
        fprintf(CASE, 'scalar per node: 1 %s %s.%s.****\n', varset{n}, basename, varset{n});
    end
end

%
%fprintf(CASE, 'scalar per node: 1 mass %s.mass.****\n', basename);
%fprintf(CASE, 'scalar per node: 1 energy %s.ener.****\n', basename);
%if isfield(frame, 'grav'); if ~isempty(frame.grav)
%    fprintf(CASE, 'scalar per node: 1 grav_potential %s.grav.****\n', basename);
%end; end
%
%fprintf(CASE, 'vector per node: 1 momentum %s.mom.****\n', basename);
%
%if ~isempty(frame.magX)
%    fprintf(CASE, 'vector per node: 1 magnet %s.mag.****\n', basename);
%end

fprintf(CASE, '\nTIME\n');
fprintf(CASE, 'time set:              1 time_data\n');
fprintf(CASE, 'number of steps:       %i\n', numel(range));
fprintf(CASE, 'filename start number: 0\n');
fprintf(CASE, 'filename increment:    1\n');
fprintf('Emitting CASE file time values...\n');
nwritten = fprintf(CASE, 'time values: ');

fprintf('Writing time meta: ');
for q = 1:numel(range)
    m = SP.getMetadata(range(q));
    tau = sum(m.time.history);

    nwritten = nwritten + fprintf(CASE,'%5.5g ', tau/timeNormalization);
    if nwritten > 72; fprintf(CASE, '\n'); nwritten = 0; end
    fprintf('%i ', int32(q));
end
fprintf(CASE, '\n');
fclose(CASE);

fprintf('\n');

% Don't want this unless doing parallel data apparently.
%fprintf(CASE, '\n\nFILE\n');
%fprintf(CASE, 'file set: 1\n');
%fprintf(CASE, 'number of steps: %i\n', numel(range)); 

end
