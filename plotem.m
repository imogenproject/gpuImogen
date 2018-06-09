% This is a terrible function mainly meant to provide instant feedback during debugging

p2 = 3;

if exist('run','var')
    
    if exist('fluidno','var')
        fn = fluidno;
    else
        fn = 1;
    end
    
    thingA = run.fluid(fn).mom(1).array./run.fluid(fn).mass.array;
    thingB = run.fluid(fn).mom(2).array./run.fluid(fn).mass.array;
    
    thingA = log10(abs(thingA));
    thingB = log10(abs(thingB));
    
    if numel(find(size(run.fluid(fn).mass.array) > 1)) > 1 % 2d
        
        if numel(find(size(run.fluid(fn).mass.array) > 1)) > 2 % 3d
            if exist('zslice')
                z0 = zslice;
            else
                z0 = round(size(run.fluid(fn).mass.array,3)/2);
            end
            eint = run.fluid(fn).ener.array(:,:,z0) - .5*(run.fluid(fn).mom(1).array(:,:,z0).^2+run.fluid(fn).mom(2).array(:,:,z0).^2+run.fluid(fn).mom(3).array(:,:,z0).^2)./run.fluid(fn).mass.array(:,:,z0);
            subplot(2,2,3); imagesc(squeeze(log10(run.fluid(fn).mass.array(:,:,z0)))); title('log10[mass]'); colorbar;
            subplot(2,2,1); imagesc(squeeze(thingA(:,:,z0))); title('v(1)'); colorbar;
            subplot(2,2,2); imagesc(squeeze(thingB(:,:,z0))); title('v(2)'); colorbar;
            subplot(2,2,4); imagesc(squeeze(eint./run.fluid(fn).mass.array(:,:,z0))); title('T'); colorbar;
        else
            eint = run.fluid(fn).ener.array - .5*(run.fluid(fn).mom(1).array.^2+run.fluid(fn).mom(2).array.^2+run.fluid(fn).mom(3).array.^2)./run.fluid(fn).mass.array;
            subplot(2,2,3); imagesc(squeeze(log10(run.fluid(fn).mass.array))); title('log10[mass]'); colorbar;
            subplot(2,2,1); imagesc(squeeze(thingA)); title('v(1)'); colorbar;
            subplot(2,2,2); imagesc(squeeze(thingB)); title('v(2)'); colorbar;
            subplot(2,2,4); imagesc(squeeze(eint./run.fluid(fn).mass.array)); title('T'); colorbar;
        end
    else
        if numel(run.fluid) > 1; MF = 1; else; MF = 0; end
        
        if MF; for f = 1:4; subplot(2,2,f); hold off; end; end
        
        for f = 1:numel(run.fluid)
            S = run.fluid(f); % S for species
            eint = S.ener.array - .5*(S.mom(1).array.^2+S.mom(2).array.^2+S.mom(3).array.^2)./S.mass.array;
            
            subplot(2,2,3); plot(log10(S.mass.array)); title('log10[mass]');
            subplot(2,2,1); plot(S.mom(1).array./S.mass.array); title('v(1)');
            subplot(2,2,2); plot(S.mom(2).array./S.mass.array); title('v(2)');
            subplot(2,2,4); plot(eint./S.mass.array); title('Temperature');
            if MF && (f == 1); for g = 1:4; subplot(2,2,g); hold on; end; end
        end
    end
    
else
    
    eint = ener.array - .5*(mom(1).array.^2+mom(2).array.^2+mom(3).array.^2)./mass.array;
    
    if numel(find(size(mass.array) > 1)) > 1 % 2d
        subplot(2,2,3); imagesc(log10(mass.array)); title('log10 mass'); colorbar;
        subplot(2,2,1); imagesc(mom(1).array./mass.array); title('v(1)'); colorbar;
        subplot(2,2,2); imagesc(mom(p2).array./mass.array); title('v(2)'); colorbar;
        subplot(2,2,4); imagesc(log10(eint)); title('log10[Einternal]'); colorbar;
    else
        subplot(2,2,3); plot(log10(mass.array)); title('mass');
        subplot(2,2,1); plot(mom(1).array); title('mom(1)');
        subplot(2,2,2); plot(mom(2).array); title('mom(2)');
        subplot(2,2,4); plot(eint./mass.array); title('T');
        
    end
    
end
