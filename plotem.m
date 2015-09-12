subplot(2,2,3); imagesc(mass.array); title('mass'); colorbar
subplot(2,2,1); imagesc(mom(1).array); title('mom(1)'); colorbar
subplot(2,2,2); imagesc(mom(2).array); title('mom(2)'); colorbar;
eint = ener.array - .5*(mom(1).array.^2+mom(2).array.^2+mom(3).array.^2)./mass.array;
subplot(2,2,4); imagesc(.6666*eint); title('Pressure'); colorbar;

