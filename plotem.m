% This is a terrible function mainly meant to provide instant freedback during debugging

eint = ener.array - .5*(mom(1).array.^2+mom(2).array.^2+mom(3).array.^2)./mass.array;

if numel(find(size(mass.array) > 1)) > 1 % 2d
    subplot(2,2,3); imagesc(mass.array); title('mass'); colorbar;
    subplot(2,2,1); imagesc(mom(1).array); title('mom(1)'); colorbar;
    subplot(2,2,2); imagesc(mom(2).array); title('mom(2)'); colorbar;
    subplot(2,2,4); imagesc(.6666*eint); title('Pressure'); colorbar;
else
    subplot(2,2,3); plot(mass.array); title('mass');
    subplot(2,2,1); plot(mom(1).array); title('mom(1)');
    subplot(2,2,2); plot(mom(2).array); title('mom(2)');
    subplot(2,2,4); plot(.6666*eint); title('Pressure');
end

