function makeEnsightScalarFile(filebase, M, vardesc, reverseIndexOrder)

SCAL = fopen(filebase, 'w');

% Write description & 'part'
charstr = char(32*ones([160 1]));
charstr(1:length(vardesc)) = vardesc;
charstr(81:84)             = 'part';
fwrite(SCAL, charstr, 'char*1');

% Write part number
fwrite(SCAL, 1, 'int');

% write 'block'
charstr = char(32*ones([80 1]));
charstr(1:5) = 'block';
fwrite(SCAL, charstr, 'char*1');

% write all scalars in array M
if reverseIndexOrder
    fwrite(SCAL, reshape(single(permute(M, [3 2 1])), [numel(M) 1]), 'float');
else
    fwrite(SCAL, reshape(single(M), [numel(M) 1]), 'float');
end

fclose(SCAL);
end
