function exportFrameToEnsight(frame)
% FIXME: Add stuff to support not loading everything first

basename = input('Basename: ','s');

exportEnsightDatafiles(basename, 0, frame);
writeEnsightMasterFiles(basename, frame.time.time, frame, 1);

end
