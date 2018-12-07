function util_delayProptoRank(t)
% Pauses for t * my rank; Useful for making rapid-fire debug output appear in order

x = mpi_basicinfo();

pause(t*x(2));

end

