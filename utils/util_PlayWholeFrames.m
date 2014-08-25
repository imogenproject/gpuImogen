function util_playWholeFrame(basename, padding, framerange, arraytype, speed)
%function util_playWholeFrame(basename, padding, framerange, precise, arraytype, speed)
% basename = '1D_XY' or something similar
% padding = The number of digits in the data enumeration
% framerange = A matrix containing the numbers of the frames you want to see. Ex: [0:20:400]
% arraytype = The data you want to see. Ex: 'mass', 'momX', 'ener'
% speed = The time in seconds to wait in between frames

for i = 1:numel(framerange)
framenum = framerange(i);
f = util_LoadWholeFrame(basename,padding,framenum); 
X = getfield(f,arraytype);
plot(X(:,1)); pause(speed)
end
end
