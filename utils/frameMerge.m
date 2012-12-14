function massiveFrame = frameMerge(prefix, padding, framenum)

f0 = frameNumberToData(prefix, padding, 0, framenum); % We need one for reference

massiveFrame = f0;
globalRes = f0.parallel.globalDims;

massiveFrame.mass = zeros(globalRes);
massiveFrame.momX = zeros(globalRes);
massiveFrame.momY = zeros(globalRes);
massiveFrame.momZ = zeros(globalRes);
massiveFrame.ener = zeros(globalRes);
if numel(f0.magX) > 0
    massiveFrame.magX = zeros(globalRes);
    massiveFrame.magY = zeros(globalRes);
    massiveFrame.magZ = zeros(globalRes);
end

%ranks = f0.parallel.geometry;
ranks = [0 1 2;3 4 5;6 7 8];
fieldset = {'mass','momX','momY','momZ','ener'};
bset     = {'magX','magY','magZ'};

  for u = 1:numel(ranks)
    frame = frameNumberToData(prefix, padding, ranks(u), framenum);
    fs = size(frame.mass); if numel(fs) == 2; fs(3) = 1;  end
    rs = size(ranks); if numel(rs) == 2; rs(3) = 1; end
    frmsize = fs - 6*(rs > 1);
    if numel(frmsize) == 2; frmsize(3) = 1; end

    frmset = {frame.parallel.myOffset(1)+(1:frmsize(1)), ...
              frame.parallel.myOffset(2)+(1:frmsize(2)), ...
              frame.parallel.myOffset(3)+(1:frmsize(3))}

    massiveFrame.mass(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.mass, ranks);
    massiveFrame.momX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momX, ranks);
    massiveFrame.momY(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momY, ranks);
    massiveFrame.momZ(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.momZ, ranks);
    massiveFrame.ener(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.ener, ranks);
    if numel(f0.magX) > 0
        massiveFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks);
        massiveFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks);
        massiveFrame.magX(frmset{1},frmset{2},frmset{3}) = trimHalo(frame.magX, ranks);
    end
  end

end

function y = trimHalo(x, nprocs)
  y=x;
  if size(nprocs,1) > 1; y = x(4:(end-3),:,:); end
  if size(nprocs,2) > 1; y = y(:,4:(end-3),:); end
  if size(nprocs,3) > 1; y = y(:,:,4:(end-3)); end
end


function dataframe = frameNumberToData(namebase, padsize, rank, frameno)
    % Take first guess; Always replace _START
    fname = sprintf('%s_rank%i_%0*i.mat', namebase, rank, padsize, frameno);
    if frameno == 0; fname = sprintf('%s_rank%i_START.mat', namebase,rank); end

    % Check existance; if fails, try _FINAL then give up
    if exist(fname, 'file') == 0
        fname = sprintf('%s_rank%i_FINAL.mat', namebase,rank);
        if exist(fname, 'file') == 0
            % Weird shit is going on. Run away!
            error(sprintf('FATAL on %s: File for frame %s_%0*i existed at check time, now isn''t reachable/existent. Wtf?', ...
                           getenv('HOSTNAME'),namebase,padsize,frameno) );
        end
    end

    % Load the next frame into workspace; Assign it to a standard variable name.
    try
        tempname = load(fname);
        nom_de_plume = fieldnames(tempname);
        dataframe = getfield(tempname,nom_de_plume{1});
    catch ERR
        fprintf('SERIOUS on %s: Frame %s in cwd %s exists but load returned error:\n', getenv('HOSTNAME'), fname, pwd());
        ERR
        dataframe = -1;
    end

end

