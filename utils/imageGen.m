function imageGen(prefix, ranks, padding, set, print)

if nargin < 5; print = 0; end

nprocessed = 0;
for n = set

  parts = [];

  for j = 1:numel(ranks)
    frame = frameNumberToData(prefix, padding, ranks(j), n);
    parts{j} = frame.mass(:,:,end/2);
    fprintf('*');
  end
  fprintf('\n');

  mergedH = [];

  for k = 1:size(ranks,2);
    mergedV = [];
    for j = 1:size(ranks,1);
      tmp = parts{ j+size(ranks,1)*(k-1)  };
      % Trim the boundary exchanges off
      if j > 1;             tmp=tmp(4:end,:); end
      if j < size(ranks,1); tmp=tmp(1:(end-3),:); end
      if k > 1;             tmp=tmp(:,4:end); end
      if k < size(ranks,2); tmp=tmp(:,1:(end-3)); end

      mergedV = [mergedV; tmp ];
    end

    mergedH = [mergedH mergedV];
  end


  mergedH = (mergedH - min(min(mergedH)))./(max(max(mergedH)) - min(min(mergedH)));

  outname = sprintf('%s_%0*i.png', prefix, padding, n);
  imwrite(mergedH', outname, 'png');

  nprocessed = nprocessed + 1;

  if print
    if mod(nprocessed,50) == 0; fprintf('\n'); end
    fprintf('X'); 
  end
end

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

