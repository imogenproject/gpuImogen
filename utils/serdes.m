classdef serdes < handle

properties (Constant = true, GetAccess = Private)
	STRUCTMARKER = int32(1400132421);
	ARRAYMARKER = int32(1145131845);
	% Note to self: add cells here some day. *Yawn*
end

methods (Access = public)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write/read one array of arbitrary size (up to 4B elements) to file
    function writeNDArray(self, FILE, array)
        % Drop a short, sweet metadata tag
        marker  = ARRAYMARKER; % ascii for ESAD, Erik's Serializer And Deserializer
        dim     = size(array);
        nd      = numel(dim);
        itsreal = isreal(array);
        writetype = [];
        if isa(array, 'integer'); vsize = 0; writetype = 'int32'; end;
	if isa(array, 'single'); vsize = 1; writetype = 'float'; end;
	if isa(array, 'double'); vsize = 2; writetype = 'double'; end

        metatag = int32([marker nd dim itsreal vsize]);

        fwrite(FILE, metatag, 'int32');
        fwrite(FILE, real(array(:)), writetype);
        if itsreal == 0
            fwrite(FILE, imag(array(:)), writetype);
        end

    end

    function array = readNDArray(self, FILE)
        marker = fread(FILE, 1, 'int32');
        if marker ~= ARRAYMARKER
            fseek(FILE, -4, 0); % rewind, this was not expected
            error('serdes error: FILE not positioned at the start of an array I wrote.\n');
            array = []; return;
        end

        nd      = fread(FILE, 1, 'int32');
        dim     = fread(FILE, nd, 'int32')';
        itsreal = fread(FILE, 1, 'int32');
        vsize   = fread(FILE, 1, 'int32');
    
	if vsize == 0; readtype = 'int32';
	if vsize == 1; readtype = 'float';
	if vsize == 2; readtype = 'double';

        array = fread(FILE, prod(dim), readtype);
        if itsreal == 0
            array = array + 1i*fread(FILE, prod(dim), readtype);
        end
        array = reshape(array, dim);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write/read a structure of potentially arbitrary complexity to file
    function writeStructure(self, FILE, struct)
        names = fieldnames(struct);
        narrays = numel(names);

        fwrite(FILE, STRUCTMARKER, 'int32'); % EStS in ascii - Erik's Structure Serializer

        % write number of arrays, followed by [strlen(string) string array]
        fwrite(FILE, int32(narrays), 'int32');
        for x = 1:narrays
	    fwrite(FILE, int32(numel(names{x})), 'int32');
            fwrite(FILE, char(names{x}), 'char*1');

            self.writeNextField(FILE, getfield(struct, names{x}));
        end
        
    end

    function struct = readStructure(self, FILE)
        struct = [];
	marker = fread(FILE, 1, 'int32');
        if marker ~= STRUCTMARKER
	    error('serdes error: FILE not positioned at the start of a struct I wrote.\n');
        end

	narrays = fread(FILE, 1, 'int32');
	for x = 1:narrays
	    slen = fread(FILE, 1, 'int32');
	    name = char(fread(FILE, slen, 'char*1')');
	    struct = setfield(struct, name, self.readNextField(FILE));
	end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Wrapper abstraction enables recursive serialization

    function writeNextField(self, FILE, field)
        if isa(field, 'struct'); self.writeStructure(FILE, field); return; end;
	if isa(field, 'float') || isa(field, 'integer'); self.writeNDArray(FILE, field); return; end;
    end

    function result = readNextField(self, FILE)
	% This pulls the next field marker then backs up
        marker = fread(FILE, 1, 'int32');
        fseek(FILE, -4, 0);

        if marker == STRUCTMARKER; result = self.readStructure(FILE); return; end
        if marker == ARRAYMARKER;  result = self.readNDArray(FILE); return; end

        error('serdes error: readNextField called but FILE not positioned at any known marker.\n');
    end

end

end
