function info = savefileInfo(pathname)

info.original = pathname;
info.path = fileparts(pathname);
k = 1; y = []; for j = (numel(fileparts(pathname))+2):numel(pathname); y(k)=pathname(j); k=k+1;end
y=char(y);

j = 1; while y(j) ~= '_'; j = j+1; end; j=j+1; while y(j) ~= '_'; j = j+1; end; j

info.prefix = y(1:j);

ints = sscanf(y((j+1):end), 'rank%i_%i');
info.frameno = ints(2);
info.rank = ints(1);

y=y((j+1):end);
j = 1; while y(j) ~= '_'; j = j+1; end
y=y(j:end);

if strcmp( y((end-2):end) ,'mat') == true
  info.pad = numel(y) - 5;
  info.extension = 'mat';
else
  info.pad = numel(y) - 4;
  info.extension = 'nc';
end


end
