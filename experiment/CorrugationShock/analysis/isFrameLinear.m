function L = isFrameLinear(framedt)

% Reject early-time relaxation
if numel(framedt) < 1001; L = 0; return; end

N = numel(framedt);

tau = framedt((end-1000):end); % grab the last 1000 timesteps

tauf = abs(fft(tau));
tauf = tauf(2:end); % drop k=0 mode

%figure(3); plot(tauf);
q = numel(find(tauf > .3*max(tauf(100:500))));


L = (q < 25);

end
