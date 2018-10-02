classdef LinkedListNode < handle
    % Class annotation template for creating new classes.
    %___________________________________________________________________________________________________
    
    %===================================================================================================
    properties (Constant = true, Transient = true) %                 C O N S T A N T         [P]
    end%CONSTANT
    
    %===================================================================================================
    properties (SetAccess = private, GetAccess = public) %                           P U B L I C  [P]
	Prev;
	Next;
    end %PUBLIC
    
    %===================================================================================================
    properties (SetAccess = public, GetAccess = public) %                P R O T E C T E D [P]
	whatami;
    end %PROTECTED
    
    %===================================================================================================
    methods %                                                                     G E T / S E T  [M]
	function node = LinkedListNode()
		node.whatami = 'UnnamedLLN';
	end

	function delete(node)
		unlink(node);
	end

	function unlink(node)
	        % Helper for if debugging
                %disp(['A ' node.whatami ' is dying.']);
		p = node.Prev;
		n = node.Next;
		
		if ~isempty(p)
			p.Next = n;
		end
		if ~isempty(n)
			n.Prev = p;
		end
		% Delete internal refs
		node.Next = [];
		node.Prev = [];	
	end

	function insertBefore(self, before)
		unlink(self);
		self.Prev = before.Prev;
		if ~isempty(self.Prev)
			self.Prev.Next = self;
		end
		self.Next = before;
		before.Prev = self;
	end

	function insertAfter(self, node)
		unlink(self);
		self.Prev = node;
		self.Next = node.Next;
		self.Prev.Next = self;
		if ~isempty(self.Next)
			self.Next.Prev = self;
		end
    end

    function f = findThe(self, name)
        if isa(self, name)
            f = self;
        else 
            if ~isempty(self.Next)
                f = self.Next.findThe(name);
            else
                f = [];
            end
        end
    end
    
    end%GET/SET
    
    %===================================================================================================
    methods (Access = public) %                                                     P U B L I C  [M]
    end%PUBLIC
    
    %===================================================================================================
    methods (Access = protected) %                                      P R O T E C T E D    [M]
    end%PROTECTED
    
    %===================================================================================================
    methods (Static = true) %                                                 S T A T I C    [M]
    end%PROTECTED
    
end%CLASS
