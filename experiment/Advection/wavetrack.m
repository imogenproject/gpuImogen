function amp = wavetrack(frame, kx0, ky0, modes)

u = fft2(frame.mass);

amp = [];

for n = 1:modes
    amp(n) = u(1+n*kx0, 1+n*ky0);
end

end
