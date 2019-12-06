function x = util_inputNumberBulletproof(query, def, tries)
% x = util_inputNumberBulletproof(query, default, tries)
% uses input(query) to try and get a number.
% There are four combinations of the second two arguments:
% neither      - continues to prompt for a valid numeric input until it gets it
% default only - returns default if input is [].
% tries only   - retries for 'tries' attempts, then returns 0
% both         - prompts for up to 'tries' attempts, but stops and returns default if input is blank

x = [];

ntries = 0;
if nargin < 3; tries = Inf; end
if nargin < 2; def = 0; havedef = 0; else; havedef = 1; end

while 1
    y = input(query, 's');
    x = str2num(y);
    if isnumeric(x) && ~isempty(x); break; end

    if isempty(x) && havedef; x = def; break; end;

    ntries = ntries + 1;
    if ntries >= tries
        if havedef; x = def; else; x = 0; end
        break;
    end

    fprintf('Invalid input; Please provide input that evalutes to a number\n');
end

end
