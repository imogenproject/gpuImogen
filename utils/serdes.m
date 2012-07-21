classdef serdes < handle

methods (Access = public)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write/read one array of arbitrary size (up to 4B elements) to file
    function writeNDArray(self, FILE, array)
        % Drop a short, sweet metadata tag
        marker  = int32(1145131845); % ascii for ESAD, Erik's Serializer And Deserializer
        dim     = size(array);
        nd      = numel(dim);
        itsreal = isreal(array);
        vsize   = isa(array,'double');

        metatag = int32([marker nd dim itsreal vsize]);

        if vsize == 1; writetype = 'double'; else; writetype = 'float'; end

        fwrite(FILE, metatag, 'int32');
        fwrite(FILE, real(array(:)), writetype);
        if itsreal == 0
            fwrite(FILE, imag(array(:)), writetype);
        end

    end

    function array = readNDArray(self, FILE)
        marker = fread(FILE, 1, 'int32');
        if marker ~= int32(1145131845)
            fseek(FILE, -4, 0); % rewind, this was not expected
            error('serdes error: FILE not positioned at the start of an array I wrote.\n');
            array = []; return;
        end

        nd      = fread(FILE, 1, 'int32');
        dim     = fread(FILE, nd, 'int32')';
        itsreal = fread(FILE, 1, 'int32');
        vsize   = fread(FILE, 1, 'int32');
    
        if vsize == 1; readtype = 'double'; else; readtype = 'float'; end

        array = fread(FILE, prod(dim), readtype);
        if itsreal == 0
            array = array + 1i*fread(FILE, prod(dim), readtype);
        end
        array = reshape(array, dim);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write/read all the arrays in a structure to file

    function writeStructsArrays(self, FILE, struct)
        names = fieldnames(struct);
        narrays = numel(names);

        fwrite(FILE, int32(1400132421), 'int32'); % EStS in ascii - Erik's Structure Serializer
        % write number of arrays, followed by [strlen(string) string array]

        fwrite(FILE, int32(narrays), 'int32');
        for x = 1:narrays
	    fwrite(FILE, int32(numel(names{x})), 'int32');
            fwrite(FILE, char(names{x}), 'char*1');
            self.writeNDArray(FILE, getfield(struct,names{x}));
        end
        
    end

    function struct = readStructsArrays(self, FILE)
        struct = [];
	marker = fread(FILE, 1, 'int32');
        if marker ~= int32(1400132421)
	    error('serdes error: FILE not positioned at the start of a struct I wrote.\n');
        end

	narrays = fread(FILE, 1, 'int32');
	for x = 1:narrays
	    slen = fread(FILE, 1, 'int32');
	    name = char(fread(FILE, slen, 'char*1')');
	    struct = setfield(struct, name, self.readNDArray(FILE));
	end
    end
end

end
