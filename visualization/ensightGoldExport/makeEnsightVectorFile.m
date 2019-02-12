function makeEnsightVectorFile(filebase, Mx, My, Mz, vardesc, reverseIndexOrder)

VEC = fopen(filebase, 'w');

% Write description & 'part'
charstr = char(32*ones([160 1]));
charstr(1:length(vardesc)) = vardesc;
charstr(81:84)             = 'part';
fwrite(VEC, charstr, 'char*1');

% Write part number
fwrite(VEC, 1, 'int');

% write 'block'
charstr = char(32*ones([80 1]));
charstr(1:5) = 'block';
fwrite(VEC, charstr, 'char*1');

if reverseIndexOrder
    % write all x-direction, ydirection and zdirection vectors
    fwrite(VEC, reshape(single(permute(Mx, [3 2 1])), [numel(Mx) 1]), 'float');
    fwrite(VEC, reshape(single(permute(My, [3 2 1])), [numel(My) 1]), 'float');
    fwrite(VEC, reshape(single(permute(Mz, [3 2 1])), [numel(Mz) 1]), 'float');
else
    fwrite(VEC, reshape(single(Mx), [numel(Mx) 1]), 'float');
    fwrite(VEC, reshape(single(My), [numel(My) 1]), 'float');
    fwrite(VEC, reshape(single(Mz), [numel(Mz) 1]), 'float');
end
fclose(VEC);

end
